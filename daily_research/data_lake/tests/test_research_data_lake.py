import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd

from daily_research.data_lake import ResearchDataLake, build_label_completeness_summary
from daily_research.data_lake import build_pool_view
from daily_research.data_lake import build_sector_board_view
from daily_research.data_lake import build_research_database
from daily_research.data_lake.pool_views import PoolViewSpec, build_pool_view_from_policy_bundle, load_pool_view
from daily_research.data_lake.sector_board_views import (
    SectorBoardViewSpec,
    build_sector_board_view_from_policy_bundle,
    load_sector_board_view,
)
from daily_research.data_lake.import_legacy_training_caches import import_legacy_training_dataset_caches
from daily_research.data_lake.policy_input_loader import load_policy_inputs_from_lake
from daily_research.data_lake.policy_input_audit import audit_policy_input_bundle
from daily_research.continuous_policy.state_builder import prepare_policy_inputs
from daily_research.continuous_policy.training_dataset_cache import save_training_dataset_cache


def _synthetic_policy_bundle_parts(
    dates: pd.DatetimeIndex,
    stocks: list[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, pd.DataFrame]]:
    resolved_stocks = stocks or ["000001.SZ", "600000.SH"]
    close = pd.DataFrame(
        [[10.0 + row + col for col in range(len(resolved_stocks))] for row in range(len(dates))],
        index=dates,
        columns=resolved_stocks,
    )
    market_frames = {
        "Open": close - 0.1,
        "High": close + 0.2,
        "Low": close - 0.2,
        "Close": close,
        "Volume": pd.DataFrame(1000.0, index=dates, columns=resolved_stocks),
        "Amount": pd.DataFrame(10000.0, index=dates, columns=resolved_stocks),
    }
    membership_frame = pd.DataFrame(True, index=dates, columns=resolved_stocks)
    feature_frames = {
        "score_none": close * 0.0,
        "score_v2": close * 0.0 + 0.1,
        "score_blend": close * 0.0 + 0.05,
    }
    return market_frames, membership_frame, feature_frames


class ResearchDataLakeTest(unittest.TestCase):
    def test_catalog_round_trips_training_dataset_and_reuses_fingerprint(self) -> None:
        sample_frame = pd.DataFrame(
            {
                "date": ["2026-01-05", "2026-01-05"],
                "stock": ["000001.SZ", "600000.SH"],
                "action_label": ["reduce", "open"],
                "current_weight": [0.20, 0.0],
                "fwd_excess_20d": [0.03, 0.05],
            }
        )
        daily_frame = pd.DataFrame({"date": ["2026-01-05"], "gross_exposure_target": [0.80]})
        spec = {
            "dataset": "continuous_policy_training_matrices",
            "pool_name": "learned_all_a",
            "benchmark": "000300.SH",
            "start_date": "20260105",
            "end_date": "20260105",
            "label_preset": "holdcash_v3",
            "max_universe_size": 1200,
        }
        label_summary = {
            "zone": "strict_train",
            "max_forward_horizon": 20,
            "observed_label_row_ratio": 1.0,
            "unobserved_label_rows": 0,
            "strict_end_date": "2026-01-05",
        }

        with TemporaryDirectory() as temp_dir:
            lake = ResearchDataLake(Path(temp_dir))
            saved = lake.save_training_dataset(
                spec=spec,
                sample_frame=sample_frame,
                daily_frame=daily_frame,
                teacher_summary={"label_preset": "holdcash_v3"},
                zone="strict_train",
                label_completeness_summary=label_summary,
            )
            reused = lake.save_training_dataset(
                spec=spec,
                sample_frame=sample_frame,
                daily_frame=daily_frame,
                teacher_summary={"label_preset": "holdcash_v3"},
                zone="strict_train",
                label_completeness_summary=label_summary,
            )
            loaded = lake.load_training_dataset(saved.dataset_id)
            listed = lake.list_datasets()
            described = lake.describe_dataset(saved.dataset_id)

        self.assertEqual(reused.dataset_id, saved.dataset_id)
        self.assertEqual(reused.status, "hit")
        self.assertEqual(len(listed), 1)
        self.assertEqual(described["dataset_id"], saved.dataset_id)
        self.assertEqual(described["zone"], "strict_train")
        self.assertEqual(described["label_completeness_summary"]["observed_label_row_ratio"], 1.0)
        pd.testing.assert_frame_equal(loaded.sample_frame.reset_index(drop=True), sample_frame)
        pd.testing.assert_frame_equal(loaded.daily_frame.reset_index(drop=True), daily_frame)
        self.assertEqual(loaded.teacher_summary["label_preset"], "holdcash_v3")

    def test_market_bundle_writes_bronze_and_silver_tables_queryable_by_duckdb(self) -> None:
        dates = pd.to_datetime(["2026-01-05", "2026-01-06"])
        stocks = ["000001.SZ", "600000.SH"]
        close = pd.DataFrame([[10.0, 20.0], [10.5, 19.8]], index=dates, columns=stocks)
        market_frames = {
            "Open": close - 0.1,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": pd.DataFrame([[1000, 2000], [1100, 2100]], index=dates, columns=stocks),
            "Amount": pd.DataFrame([[10000, 40000], [11550, 41580]], index=dates, columns=stocks),
        }
        benchmark_close = pd.Series([4000.0, 4010.0], index=dates, name="000300.SH")
        membership_frame = pd.DataFrame(
            {
                "date": ["2026-01-05", "2026-01-06"],
                "stock": ["000001.SZ", "600000.SH"],
                "in_pool": [1.0, 1.0],
            }
        )
        feature_frames = {
            "score_blend": pd.DataFrame([[0.2, 0.5], [0.3, 0.4]], index=dates, columns=stocks),
        }

        with TemporaryDirectory() as temp_dir:
            lake = ResearchDataLake(Path(temp_dir))
            record = lake.save_market_data_bundle(
                spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
                market_frames=market_frames,
                benchmark_close=benchmark_close,
                membership_frame=membership_frame,
                feature_frames=feature_frames,
                source="synthetic",
            )
            price_count = lake.query(
                "select count(*) as n from read_parquet(?)",
                [record.content_paths["bronze_market_data"]],
            )["n"].iloc[0]
            feature_panel_rows = lake.query(
                "select count(*) as n from read_parquet(?)",
                [record.content_paths["silver_feature_values"]],
            )["n"].iloc[0]
            membership_exists = Path(record.content_paths["silver_membership"]).exists()

        self.assertEqual(int(price_count), 4)
        self.assertEqual(int(feature_panel_rows), 2)
        self.assertEqual(record.row_counts["bronze_market_data"], 4)
        self.assertEqual(record.row_counts["silver_feature_panels"], 1)
        self.assertEqual(record.row_counts["silver_feature_cells"], 4)
        self.assertTrue(membership_exists)

    def test_market_bundle_can_be_loaded_as_policy_inputs_without_tq(self) -> None:
        dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        stocks = ["000001.SZ", "600000.SH"]
        close = pd.DataFrame([[10.0, 20.0], [10.5, 19.8], [10.7, 20.2]], index=dates, columns=stocks)
        market_frames = {
            "Open": close - 0.1,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": pd.DataFrame(1000.0, index=dates, columns=stocks),
            "Amount": pd.DataFrame(10000.0, index=dates, columns=stocks),
        }
        benchmark_close = pd.Series([4000.0, 4010.0, 4020.0], index=dates, name="000300.SH")
        membership_frame = pd.DataFrame(True, index=dates, columns=stocks)
        feature_frames = {
            "score_none": close * 0.0,
            "score_v2": close * 0.0 + 0.1,
            "score_blend": close * 0.0 + 0.05,
            "ret_1d": close.pct_change(),
            "score_delta_1d": close * 0.0,
            "score_blend_lag1": close.shift(1) * 0.0,
        }

        with TemporaryDirectory() as temp_dir:
            lake = ResearchDataLake(Path(temp_dir))
            record = lake.save_market_data_bundle(
                spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
                market_frames=market_frames,
                benchmark_close=benchmark_close,
                membership_frame=membership_frame,
                feature_frames=feature_frames,
                source="synthetic",
            )
            prepared = load_policy_inputs_from_lake(
                lake=lake,
                dataset_id=record.dataset_id,
                start_date="2026-01-05",
                end_date="2026-01-07",
            )

        self.assertEqual(prepared.universe, tuple(stocks))
        pd.testing.assert_frame_equal(prepared.close, close)
        self.assertEqual(prepared.raw_cache_meta["source"], "data_lake")
        self.assertIn("score_blend_lag1", prepared.derived_frames)

    def test_lake_loader_derives_state_frames_from_market_data(self) -> None:
        dates = pd.date_range("2026-01-05", periods=30, freq="B")
        market_frames, membership_frame, feature_frames = _synthetic_policy_bundle_parts(dates)
        close = market_frames["Close"]
        benchmark_close = pd.Series([4000.0 + idx for idx in range(len(dates))], index=dates, name="000300.SH")

        with TemporaryDirectory() as temp_dir:
            lake = ResearchDataLake(Path(temp_dir))
            record = lake.save_market_data_bundle(
                spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
                market_frames=market_frames,
                benchmark_close=benchmark_close,
                membership_frame=membership_frame,
                feature_frames=feature_frames,
                source="synthetic",
            )
            prepared = load_policy_inputs_from_lake(
                lake=lake,
                dataset_id=record.dataset_id,
                start_date="2026-01-05",
                end_date=dates[-1].strftime("%Y-%m-%d"),
            )

        required = {
            "ret_1d",
            "ret_3d",
            "ret_5d",
            "ret_10d",
            "ret_20d",
            "vol_5d",
            "vol_20d",
            "score_delta_1d",
            "score_delta_5d",
            "score_delta_accel",
            "ret_accel_5_20",
            "volume_ratio_5_20",
            "distance_to_20d_high",
            "distance_to_60d_high",
            "distance_to_20d_low",
            "volatility_expansion",
            "adv_ratio_5_20",
            "ret_1d_lag1",
            "vol_20d_lag4",
        }
        self.assertTrue(required.issubset(set(prepared.derived_frames)))
        signal_dt = dates[-1]
        for frame_name in required:
            prepared.derived_frames[frame_name].loc[signal_dt].reindex(prepared.universe).to_numpy(dtype=float)
        pd.testing.assert_frame_equal(
            prepared.derived_frames["ret_1d"],
            close.pct_change(fill_method=None),
            check_freq=False,
        )
        pd.testing.assert_frame_equal(
            prepared.derived_frames["score_delta_1d"],
            prepared.score_blend.diff(1),
            check_freq=False,
        )

    def test_loader_resolves_relocated_lake_content_paths(self) -> None:
        dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        market_frames, membership_frame, feature_frames = _synthetic_policy_bundle_parts(dates)
        feature_frames.update(
            {
                "ret_1d": market_frames["Close"].pct_change(),
                "score_delta_1d": market_frames["Close"] * 0.0,
                "score_blend_lag1": market_frames["Close"].shift(1) * 0.0,
            }
        )
        benchmark_close = pd.Series([4000.0, 4010.0, 4020.0], index=dates, name="000300.SH")

        with TemporaryDirectory() as temp_dir:
            lake = ResearchDataLake(Path(temp_dir))
            record = lake.save_market_data_bundle(
                spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
                market_frames=market_frames,
                benchmark_close=benchmark_close,
                membership_frame=membership_frame,
                feature_frames=feature_frames,
                source="synthetic",
            )
            metadata = lake.describe_dataset(record.dataset_id)
            stale_paths = {
                key: str(value).replace(str(Path(temp_dir)), r"H:\old_project\daily_research\output\research_data_lake")
                for key, value in metadata["content_paths"].items()
            }
            lake.query(
                "update datasets set content_paths_json = ? where dataset_id = ?",
                [json.dumps(stale_paths), record.dataset_id],
            )

            prepared = load_policy_inputs_from_lake(
                lake=lake,
                dataset_id=record.dataset_id,
                start_date="2026-01-05",
                end_date="2026-01-07",
            )

        self.assertEqual(prepared.raw_cache_meta["dataset_id"], record.dataset_id)
        pd.testing.assert_frame_equal(prepared.close, market_frames["Close"])

    def test_market_bundle_persists_benchmark_open_and_loader_uses_it(self) -> None:
        dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        market_frames, membership_frame, feature_frames = _synthetic_policy_bundle_parts(dates)
        benchmark_close = pd.Series([4000.0, 4010.0, 4020.0], index=dates, name="000300.SH")
        benchmark_open = pd.Series([3995.0, 4005.0, 4015.0], index=dates, name="000300.SH")

        with TemporaryDirectory() as temp_dir:
            lake = ResearchDataLake(Path(temp_dir))
            record = lake.save_market_data_bundle(
                spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
                market_frames=market_frames,
                benchmark_close=benchmark_close,
                benchmark_open=benchmark_open,
                membership_frame=membership_frame,
                feature_frames=feature_frames,
                source="synthetic",
            )
            saved_benchmark = pd.read_parquet(record.content_paths["silver_benchmark"])
            prepared = load_policy_inputs_from_lake(
                lake=lake,
                dataset_id=record.dataset_id,
                start_date="2026-01-05",
                end_date="2026-01-07",
                require_benchmark_open=True,
            )

        self.assertIn("open", saved_benchmark.columns)
        pd.testing.assert_series_equal(prepared.benchmark_open, benchmark_open, check_freq=False)
        coverage = prepared.raw_cache_meta["lake_coverage_report"]
        self.assertEqual(coverage["benchmark_open_source"], "silver_benchmark.open")
        self.assertEqual(coverage["benchmark_open_rows"], 3)

    def test_policy_input_loader_marks_close_fallback_and_blocks_when_open_required(self) -> None:
        dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        market_frames, membership_frame, feature_frames = _synthetic_policy_bundle_parts(dates)
        benchmark_close = pd.Series([4000.0, 4010.0, 4020.0], index=dates, name="000300.SH")

        with TemporaryDirectory() as temp_dir:
            lake = ResearchDataLake(Path(temp_dir))
            record = lake.save_market_data_bundle(
                spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
                market_frames=market_frames,
                benchmark_close=benchmark_close,
                membership_frame=membership_frame,
                feature_frames=feature_frames,
                source="synthetic",
            )
            prepared = load_policy_inputs_from_lake(
                lake=lake,
                dataset_id=record.dataset_id,
                start_date="2026-01-05",
                end_date="2026-01-07",
            )
            with self.assertRaisesRegex(ValueError, "missing_benchmark_open"):
                load_policy_inputs_from_lake(
                    lake=lake,
                    dataset_id=record.dataset_id,
                    start_date="2026-01-05",
                    end_date="2026-01-07",
                    require_benchmark_open=True,
                )

        pd.testing.assert_series_equal(prepared.benchmark_open, benchmark_close, check_freq=False)
        coverage = prepared.raw_cache_meta["lake_coverage_report"]
        self.assertEqual(coverage["benchmark_open_source"], "fallback_close")
        self.assertEqual(coverage["benchmark_open_rows"], 3)

    def test_prepare_policy_inputs_lake_uses_data_lake(self) -> None:
        dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        stocks = ["000001.SZ", "000002.SZ", "000003.SZ"]
        close = pd.DataFrame(
            [[10.0, 20.0, 30.0], [10.5, 20.5, 30.5], [11.0, 21.0, 31.0]],
            index=dates,
            columns=stocks,
        )
        market_frames = {
            "Open": close - 0.1,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": pd.DataFrame(1000.0, index=dates, columns=stocks),
            "Amount": pd.DataFrame(10000.0, index=dates, columns=stocks),
        }
        benchmark_close = pd.Series([4000.0, 4010.0, 4020.0], index=dates, name="000300.SH")
        membership_frame = pd.DataFrame(True, index=dates, columns=stocks)

        with TemporaryDirectory() as temp_dir:
            lake = ResearchDataLake(Path(temp_dir))
            record = lake.save_market_data_bundle(
                spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
                market_frames=market_frames,
                benchmark_close=benchmark_close,
                membership_frame=membership_frame,
                feature_frames={"score_none": close * 0.0, "score_v2": close * 0.0 + 0.1},
                source="synthetic",
            )
            with mock.patch(
                "daily_research.continuous_policy.state_builder.load_universe_from_tq",
                side_effect=AssertionError("lake mode must not call TDX universe resolution"),
            ):
                prepared = prepare_policy_inputs(
                    pool_name="learned_all_a",
                    start_date="2026-01-05",
                    end_date="2026-01-07",
                    benchmark="000300.SH",
                    data_source="lake",
                    lake_dataset_id=record.dataset_id,
                    data_lake_root=temp_dir,
                    max_universe_size=2,
                    extra_stocks=["000003.SZ"],
                )

        self.assertEqual(prepared.data_source, "lake")
        self.assertEqual(prepared.raw_cache_meta["dataset_id"], record.dataset_id)
        self.assertEqual(prepared.raw_cache_meta["lake_coverage_report"]["status"], "ok")
        self.assertEqual(prepared.universe, tuple(stocks))

    def test_pool_view_reuses_fingerprint_and_does_not_copy_market_data(self) -> None:
        dates = pd.date_range("2026-01-05", periods=30, freq="B")
        stocks = ["000001.SZ", "000002.SZ", "600000.SH"]
        close = pd.DataFrame(
            [[10.0 + row, 20.0 + row, 30.0 + row] for row in range(len(dates))],
            index=dates,
            columns=stocks,
        )
        amount = pd.DataFrame(
            [[10000.0 + row, 90000.0 + row, 50000.0 + row] for row in range(len(dates))],
            index=dates,
            columns=stocks,
        )
        market_frames = {
            "Open": close - 0.1,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": pd.DataFrame(1000.0, index=dates, columns=stocks),
            "Amount": amount,
        }

        with TemporaryDirectory() as temp_dir:
            lake = ResearchDataLake(Path(temp_dir))
            market_record = lake.save_market_data_bundle(
                spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
                market_frames=market_frames,
                benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
                membership_frame=pd.DataFrame(True, index=dates, columns=stocks),
                feature_frames={"score_none": close * 0.0, "score_v2": close * 0.0 + 0.1},
                source="synthetic",
            )
            spec = PoolViewSpec(
                source_market_dataset_id=market_record.dataset_id,
                view_kind="rolling_liquidity",
                view_name="rolling_liquid2",
                pool_name="liquid2",
                start_date="2026-01-05",
                end_date="2026-02-13",
                rebalance_every_days=5,
                adv_window=2,
            )
            first = build_pool_view_from_policy_bundle(lake=lake, spec=spec)
            second = build_pool_view_from_policy_bundle(lake=lake, spec=spec)
            loaded = load_pool_view(lake=lake, pool_view_id=first.dataset_id)
            quality_report_exists = Path(first.content_paths["quality_report"]).exists()

        self.assertEqual(second.dataset_id, first.dataset_id)
        self.assertEqual(second.status, "hit")
        self.assertEqual(first.dataset_kind, "policy_pool_view")
        self.assertFalse(any("bronze_market_data" in key for key in first.content_paths))
        self.assertEqual(loaded.membership_frame.shape[1], 3)
        self.assertEqual(int(loaded.membership_frame.sum(axis=1).median()), 2)
        self.assertEqual(loaded.metadata["parameters"]["source_market_dataset_id"], market_record.dataset_id)
        self.assertIn("quality_report", first.content_paths)
        self.assertEqual(loaded.metadata["source_cache"]["quality_report"]["zero_member_days"], 0)
        self.assertEqual(loaded.metadata["source_cache"]["quality_report"]["member_count_median"], 2.0)
        self.assertTrue(quality_report_exists)

    def test_pool_view_rolling_liquidity_excludes_symbol_prefixes(self) -> None:
        dates = pd.date_range("2026-01-05", periods=30, freq="B")
        stocks = ["000001.SZ", "300001.SZ", "301001.SZ", "600000.SH", "688001.SH"]
        close = pd.DataFrame(10.0, index=dates, columns=stocks)
        amount = pd.DataFrame(
            {
                "000001.SZ": [10000.0 + row for row in range(len(dates))],
                "300001.SZ": [90000.0 + row for row in range(len(dates))],
                "301001.SZ": [80000.0 + row for row in range(len(dates))],
                "600000.SH": [70000.0 + row for row in range(len(dates))],
                "688001.SH": [95000.0 + row for row in range(len(dates))],
            },
            index=dates,
        )
        market_frames = {
            "Open": close - 0.1,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": pd.DataFrame(1000.0, index=dates, columns=stocks),
            "Amount": amount,
        }

        with TemporaryDirectory() as temp_dir:
            lake = ResearchDataLake(Path(temp_dir))
            market_record = lake.save_market_data_bundle(
                spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
                market_frames=market_frames,
                benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
                membership_frame=pd.DataFrame(True, index=dates, columns=stocks),
                feature_frames={"score_none": close * 0.0, "score_v2": close * 0.0 + 0.1},
                source="synthetic",
            )
            view = build_pool_view_from_policy_bundle(
                lake=lake,
                spec=PoolViewSpec(
                    source_market_dataset_id=market_record.dataset_id,
                    view_kind="rolling_liquidity",
                    view_name="rolling_liquid2_mainboard",
                    pool_name="liquid2",
                    start_date="2026-01-05",
                    end_date="2026-02-13",
                    rebalance_every_days=5,
                    adv_window=2,
                    exclude_symbol_prefixes=("300", "301", "688", "689"),
                ),
            )
            loaded = load_pool_view(lake=lake, pool_view_id=view.dataset_id)

        self.assertEqual(set(loaded.membership_frame.columns), {"000001.SZ", "600000.SH"})
        self.assertEqual(int(loaded.membership_frame.sum(axis=1).median()), 2)
        self.assertEqual(
            loaded.metadata["parameters"]["exclude_symbol_prefixes"],
            ["300", "301", "688", "689"],
        )
        self.assertEqual(
            loaded.metadata["source_cache"]["exclude_symbol_prefixes"],
            ["300", "301", "688", "689"],
        )

    def test_pool_view_exchange_filter_and_loader_pool_view_priority(self) -> None:
        dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        stocks = ["000001.SZ", "000002.SZ", "600000.SH"]
        close = pd.DataFrame([[10.0, 20.0, 30.0], [10.5, 20.5, 30.5], [11.0, 21.0, 31.0]], index=dates, columns=stocks)
        market_frames = {
            "Open": close - 0.1,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": pd.DataFrame(1000.0, index=dates, columns=stocks),
            "Amount": pd.DataFrame(10000.0, index=dates, columns=stocks),
        }

        with TemporaryDirectory() as temp_dir:
            lake = ResearchDataLake(Path(temp_dir))
            market_record = lake.save_market_data_bundle(
                spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
                market_frames=market_frames,
                benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
                membership_frame=pd.DataFrame(True, index=dates, columns=stocks),
                feature_frames={"score_none": close * 0.0, "score_v2": close * 0.0 + 0.1},
                source="synthetic",
            )
            view = build_pool_view_from_policy_bundle(
                lake=lake,
                spec=PoolViewSpec(
                    source_market_dataset_id=market_record.dataset_id,
                    view_kind="exchange",
                    view_name="exchange_sh",
                    start_date="2026-01-05",
                    end_date="2026-01-07",
                    exchange_suffix=".SH",
                ),
            )
            prepared = load_policy_inputs_from_lake(
                lake=lake,
                dataset_id=market_record.dataset_id,
                start_date="2026-01-05",
                end_date="2026-01-07",
                universe=["000001.SZ", "000002.SZ"],
                max_universe_size=1,
                pool_view_id=view.dataset_id,
            )

        self.assertEqual(prepared.universe, ("600000.SH",))
        self.assertEqual(prepared.pool_name, "exchange_sh")
        self.assertEqual(prepared.raw_cache_meta["pool_view"]["dataset_id"], view.dataset_id)
        self.assertTrue(prepared.membership_frame.eq(True).all().all())

    def test_build_pool_view_cli_and_prepare_policy_inputs_lake_pool_view(self) -> None:
        dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        stocks = ["000001.SZ", "600000.SH"]
        close = pd.DataFrame([[10.0, 30.0], [10.5, 30.5], [11.0, 31.0]], index=dates, columns=stocks)
        market_frames = {
            "Open": close - 0.1,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": pd.DataFrame(1000.0, index=dates, columns=stocks),
            "Amount": pd.DataFrame(10000.0, index=dates, columns=stocks),
        }

        with TemporaryDirectory() as temp_dir:
            lake = ResearchDataLake(Path(temp_dir))
            market_record = lake.save_market_data_bundle(
                spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
                market_frames=market_frames,
                benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
                membership_frame=pd.DataFrame(True, index=dates, columns=stocks),
                feature_frames={"score_none": close * 0.0, "score_v2": close * 0.0 + 0.1},
                source="synthetic",
            )
            manifest = build_pool_view.main(
                [
                    "--data-lake-root",
                    temp_dir,
                    "--source-market-dataset-id",
                    market_record.dataset_id,
                    "--view-kind",
                    "exchange",
                    "--view-name",
                    "exchange_sz",
                    "--exchange-suffix",
                    ".SZ",
                    "--start-date",
                    "2026-01-05",
                    "--end-date",
                    "2026-01-07",
                ]
            )
            prepared = prepare_policy_inputs(
                pool_name="learned_all_a",
                start_date="2026-01-05",
                end_date="2026-01-07",
                benchmark="000300.SH",
                data_source="lake",
                lake_dataset_id=market_record.dataset_id,
                data_lake_root=temp_dir,
                pool_view_id=manifest["dataset_id"],
            )

        self.assertEqual(manifest["status"], "ok")
        self.assertEqual(manifest["view_kind"], "exchange")
        self.assertEqual(prepared.universe, ("000001.SZ",))
        self.assertEqual(prepared.rolling_pool_summary["pool_view_id"], manifest["dataset_id"])

    def test_load_policy_inputs_builds_pool_view_from_explicit_spec(self) -> None:
        dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        stocks = ["000001.SZ", "600000.SH"]
        close = pd.DataFrame([[10.0, 30.0], [10.5, 30.5], [11.0, 31.0]], index=dates, columns=stocks)
        market_frames = {
            "Open": close - 0.1,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": pd.DataFrame(1000.0, index=dates, columns=stocks),
            "Amount": pd.DataFrame(10000.0, index=dates, columns=stocks),
        }

        with TemporaryDirectory() as temp_dir:
            lake = ResearchDataLake(Path(temp_dir))
            market_record = lake.save_market_data_bundle(
                spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
                market_frames=market_frames,
                benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
                membership_frame=pd.DataFrame(True, index=dates, columns=stocks),
                feature_frames={"score_none": close * 0.0, "score_v2": close * 0.0 + 0.1},
                source="synthetic",
            )
            prepared = load_policy_inputs_from_lake(
                lake=lake,
                dataset_id=market_record.dataset_id,
                start_date="2026-01-05",
                end_date="2026-01-07",
                pool_view_spec={
                    "source_market_dataset_id": market_record.dataset_id,
                    "view_kind": "exchange",
                    "view_name": "exchange_sz",
                    "exchange_suffix": ".SZ",
                    "start_date": "2026-01-05",
                    "end_date": "2026-01-07",
                },
            )

        self.assertEqual(prepared.universe, ("000001.SZ",))
        self.assertTrue(prepared.raw_cache_meta["pool_view"]["dataset_id"].startswith("policy_pool_view__"))
        self.assertEqual(prepared.raw_cache_meta["pool_view"]["view_name"], "exchange_sz")

    def test_loader_rejects_pool_view_from_different_source_bundle(self) -> None:
        dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        stocks = ["000001.SZ", "600000.SH"]
        close = pd.DataFrame([[10.0, 30.0], [10.5, 30.5], [11.0, 31.0]], index=dates, columns=stocks)
        market_frames = {
            "Open": close - 0.1,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": pd.DataFrame(1000.0, index=dates, columns=stocks),
            "Amount": pd.DataFrame(10000.0, index=dates, columns=stocks),
        }

        with TemporaryDirectory() as temp_dir:
            lake = ResearchDataLake(Path(temp_dir))
            first = lake.save_market_data_bundle(
                spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic_a"},
                market_frames=market_frames,
                benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
                membership_frame=pd.DataFrame(True, index=dates, columns=stocks),
                feature_frames={"score_none": close * 0.0, "score_v2": close * 0.0 + 0.1},
                source="synthetic",
            )
            second = lake.save_market_data_bundle(
                spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic_b"},
                market_frames=market_frames,
                benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
                membership_frame=pd.DataFrame(True, index=dates, columns=stocks),
                feature_frames={"score_none": close * 0.0, "score_v2": close * 0.0 + 0.1},
                source="synthetic",
            )
            view = build_pool_view_from_policy_bundle(
                lake=lake,
                spec=PoolViewSpec(
                    source_market_dataset_id=first.dataset_id,
                    view_kind="exchange",
                    view_name="exchange_sz",
                    start_date="2026-01-05",
                    end_date="2026-01-07",
                    exchange_suffix=".SZ",
                ),
            )
            with self.assertRaisesRegex(ValueError, "pool_view_source_mismatch"):
                load_policy_inputs_from_lake(
                    lake=lake,
                    dataset_id=second.dataset_id,
                    start_date="2026-01-05",
                    end_date="2026-01-07",
                    pool_view_id=view.dataset_id,
                )

    def test_build_pool_view_cli_blocks_when_active_artifact_has_diff(self) -> None:
        with mock.patch("daily_research.data_lake.build_pool_view._active_artifact_has_diff", return_value=True):
            with self.assertRaisesRegex(ValueError, "active_artifact_diff_blocker"):
                build_pool_view.main(
                    [
                        "--source-market-dataset-id",
                        "policy_input_bundle__fixed",
                        "--view-kind",
                        "exchange",
                        "--view-name",
                        "exchange_sz",
                        "--exchange-suffix",
                        ".SZ",
                    ]
                )

    def test_sector_board_view_reuses_fingerprint_and_writes_lightweight_metadata(self) -> None:
        dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        stocks = ["000001.SZ", "600000.SH", "430001.BJ"]
        market_frames, membership_frame, feature_frames = _synthetic_policy_bundle_parts(dates, stocks=stocks)

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            industry_csv = root / "industry.csv"
            industry_csv.write_text(
                "stock,industry\n000001.SZ,银行\n600000.SH,银行\n999999.SH,缺席行业\n",
                encoding="utf-8-sig",
            )
            block_file = root / "infoharbor_block.dat"
            block_file.write_text(
                "#GN_银行,2,880001,20200101,20260519,,\n0#000001,1#600000,0#999999\n"
                "#FG_高股息,1,880002,20200101,20260519,,\n1#600000,2#430001\n",
                encoding="gbk",
            )
            lake = ResearchDataLake(root / "lake")
            market_record = lake.save_market_data_bundle(
                spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
                market_frames=market_frames,
                benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
                membership_frame=membership_frame,
                feature_frames=feature_frames,
                source="synthetic",
            )
            spec = SectorBoardViewSpec(
                source_market_dataset_id=market_record.dataset_id,
                industry_source_path=str(industry_csv),
                board_source_path=str(block_file),
                as_of_date="2026-05-19",
            )
            first = build_sector_board_view_from_policy_bundle(lake=lake, spec=spec)
            second = build_sector_board_view_from_policy_bundle(lake=lake, spec=spec)
            loaded = load_sector_board_view(lake=lake, sector_board_view_id=first.dataset_id)

        self.assertEqual(second.dataset_id, first.dataset_id)
        self.assertEqual(second.status, "hit")
        self.assertEqual(first.dataset_kind, "policy_sector_board_view")
        self.assertFalse(any("bronze_market_data" in key for key in first.content_paths))
        self.assertEqual(loaded.metadata["parameters"]["snapshot_semantics"], "latest_static_snapshot")
        self.assertEqual(loaded.metadata["source_cache"]["industry_coverage"]["covered_symbols"], 2)
        self.assertIn("430001.BJ", loaded.metadata["source_cache"]["industry_coverage"]["missing_symbols"])
        self.assertEqual(set(loaded.industry_map_frame["symbol"]), {"000001.SZ", "600000.SH"})
        self.assertEqual(set(loaded.board_membership_frame["symbol"]), {"000001.SZ", "600000.SH", "430001.BJ"})
        self.assertEqual(set(loaded.board_summary_frame["board_kind"]), {"GN", "FG"})

    def test_sector_board_view_cli_blocks_active_artifact_diff(self) -> None:
        with mock.patch("daily_research.data_lake.build_sector_board_view._active_artifact_has_diff", return_value=True):
            with self.assertRaisesRegex(ValueError, "active_artifact_diff_blocker"):
                build_sector_board_view.main(["--source-market-dataset-id", "policy_input_bundle__fixed"])

    def test_sector_board_view_uses_lake_sidecar_industry_and_explicit_empty_board(self) -> None:
        dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        stocks = ["000001.SZ", "600000.SH", "430001.BJ"]
        market_frames, membership_frame, feature_frames = _synthetic_policy_bundle_parts(dates, stocks=stocks)

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lake = ResearchDataLake(root / "lake")
            industry = pd.DataFrame(
                {
                    "symbol": ["000001.SZ", "000001.SZ", "600000.SH", "999999.SH"],
                    "trade_date": ["2026-01-05", "2026-01-07", "2026-01-06", "2026-01-06"],
                    "industry": ["旧银行", "银行", "银行", "缺席行业"],
                    "concept_tags": ["", "", "", ""],
                    "source": ["baostock"] * 4,
                }
            )
            sidecar = lake.save_domain_dataset(
                domain="industry_concept",
                frame=industry,
                spec={"dataset": "data_platform_industry_concept", "start_date": "2026-01-05", "end_date": "2026-01-07"},
                source="data_platform_refresh",
            )
            market_record = lake.save_market_data_bundle(
                spec={
                    "pool_name": "learned_all_a",
                    "benchmark": "000300.SH",
                    "source": "synthetic",
                    "sidecar_dataset_ids": {"industry_concept": sidecar.dataset_id},
                },
                market_frames=market_frames,
                benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
                membership_frame=membership_frame,
                feature_frames=feature_frames,
                source="synthetic",
            )
            sector_view = build_sector_board_view_from_policy_bundle(
                lake=lake,
                spec=SectorBoardViewSpec(
                    source_market_dataset_id=market_record.dataset_id,
                    industry_source_kind="lake_sidecar",
                    board_source_kind="empty",
                    allow_empty_board=True,
                    as_of_date="2026-01-07",
                    view_name="sector_board_baostock_industry_latest_static",
                ),
            )
            loaded = load_sector_board_view(lake=lake, sector_board_view_id=sector_view.dataset_id)

        self.assertEqual(set(loaded.industry_map_frame["symbol"]), {"000001.SZ", "600000.SH"})
        self.assertEqual(loaded.industry_map_frame.loc[loaded.industry_map_frame["symbol"].eq("000001.SZ"), "industry"].iloc[0], "银行")
        self.assertTrue(loaded.board_membership_frame.empty)
        self.assertTrue(loaded.board_summary_frame.empty)
        self.assertEqual(loaded.metadata["parameters"]["industry_source_kind"], "lake_sidecar")
        self.assertEqual(loaded.metadata["parameters"]["board_source_kind"], "empty")
        self.assertEqual(loaded.metadata["source_cache"]["industry_coverage"]["covered_symbols"], 2)
        self.assertEqual(loaded.metadata["source_cache"]["board_coverage"]["status"], "empty_explicit")

    def test_sector_board_view_rejects_empty_board_without_explicit_allowance(self) -> None:
        dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        stocks = ["000001.SZ", "600000.SH"]
        market_frames, membership_frame, feature_frames = _synthetic_policy_bundle_parts(dates, stocks=stocks)

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            industry_csv = root / "industry.csv"
            industry_csv.write_text("stock,industry\n000001.SZ,银行\n600000.SH,银行\n", encoding="utf-8-sig")
            lake = ResearchDataLake(root / "lake")
            market_record = lake.save_market_data_bundle(
                spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
                market_frames=market_frames,
                benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
                membership_frame=membership_frame,
                feature_frames=feature_frames,
                source="synthetic",
            )
            with self.assertRaisesRegex(ValueError, "empty board requires allow_empty_board"):
                build_sector_board_view_from_policy_bundle(
                    lake=lake,
                    spec=SectorBoardViewSpec(
                        source_market_dataset_id=market_record.dataset_id,
                        industry_source_path=str(industry_csv),
                        board_source_kind="empty",
                        allow_empty_board=False,
                        as_of_date="2026-01-07",
                    ),
                )

    def test_sector_board_view_cli_accepts_lake_sidecar_and_empty_board_modes(self) -> None:
        dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        stocks = ["000001.SZ", "600000.SH"]
        market_frames, membership_frame, feature_frames = _synthetic_policy_bundle_parts(dates, stocks=stocks)

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lake = ResearchDataLake(root)
            sidecar = lake.save_domain_dataset(
                domain="industry_concept",
                frame=pd.DataFrame(
                    {
                        "symbol": ["000001.SZ", "600000.SH"],
                        "trade_date": ["2026-01-07", "2026-01-07"],
                        "industry": ["银行", "银行"],
                        "concept_tags": ["", ""],
                        "source": ["baostock", "baostock"],
                    }
                ),
                spec={"dataset": "data_platform_industry_concept", "start_date": "2026-01-07", "end_date": "2026-01-07"},
                source="data_platform_refresh",
            )
            market_record = lake.save_market_data_bundle(
                spec={
                    "pool_name": "learned_all_a",
                    "benchmark": "000300.SH",
                    "source": "synthetic",
                    "sidecar_dataset_ids": {"industry_concept": sidecar.dataset_id},
                },
                market_frames=market_frames,
                benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
                membership_frame=membership_frame,
                feature_frames=feature_frames,
                source="synthetic",
            )
            manifest = build_sector_board_view.main(
                [
                    "--data-lake-root",
                    temp_dir,
                    "--source-market-dataset-id",
                    market_record.dataset_id,
                    "--industry-source-kind",
                    "lake_sidecar",
                    "--board-source-kind",
                    "empty",
                    "--allow-empty-board",
                    "--as-of-date",
                    "2026-01-07",
                    "--view-name",
                    "sector_board_baostock_industry_latest_static",
                ]
            )

        self.assertEqual(manifest["status"], "ok")
        self.assertEqual(manifest["industry_source_kind"], "lake_sidecar")
        self.assertEqual(manifest["board_source_kind"], "empty")
        self.assertEqual(manifest["board_coverage"]["status"], "empty_explicit")

    def test_lake_loader_attaches_sector_board_metadata_without_changing_universe(self) -> None:
        dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        stocks = ["000001.SZ", "600000.SH"]
        market_frames, membership_frame, feature_frames = _synthetic_policy_bundle_parts(dates, stocks=stocks)

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            industry_csv = root / "industry.csv"
            industry_csv.write_text("symbol,industry\n000001.SZ,银行\n600000.SH,银行\n", encoding="utf-8-sig")
            block_file = root / "infoharbor_block.dat"
            block_file.write_text("#GN_银行,2,880001,20200101,20260519,,\n0#000001,1#600000\n", encoding="gbk")
            lake = ResearchDataLake(root / "lake")
            market_record = lake.save_market_data_bundle(
                spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
                market_frames=market_frames,
                benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
                membership_frame=membership_frame,
                feature_frames=feature_frames,
                source="synthetic",
            )
            sector_view = build_sector_board_view_from_policy_bundle(
                lake=lake,
                spec=SectorBoardViewSpec(
                    source_market_dataset_id=market_record.dataset_id,
                    industry_source_path=str(industry_csv),
                    board_source_path=str(block_file),
                    as_of_date="2026-05-19",
                ),
            )
            pool_view = build_pool_view_from_policy_bundle(
                lake=lake,
                spec=PoolViewSpec(
                    source_market_dataset_id=market_record.dataset_id,
                    view_kind="exchange",
                    view_name="exchange_sz",
                    start_date="2026-01-05",
                    end_date="2026-01-07",
                    exchange_suffix=".SZ",
                ),
            )
            prepared = load_policy_inputs_from_lake(
                lake=lake,
                dataset_id=market_record.dataset_id,
                start_date="2026-01-05",
                end_date="2026-01-07",
                pool_view_id=pool_view.dataset_id,
                sector_board_view_id=sector_view.dataset_id,
            )

        self.assertEqual(prepared.universe, ("000001.SZ",))
        self.assertIn("industry_map", prepared.metadata_frames)
        self.assertIn("board_membership", prepared.metadata_frames)
        self.assertIn("board_summary", prepared.metadata_frames)
        self.assertEqual(prepared.raw_cache_meta["sector_board_view"]["dataset_id"], sector_view.dataset_id)
        self.assertEqual(prepared.metadata_summary["sector_board_view"]["snapshot_semantics"], "latest_static_snapshot")
        self.assertEqual(prepared.metadata_frames["industry_map"]["symbol"].tolist(), ["000001.SZ"])

    def test_lake_loader_rejects_sector_board_view_from_different_bundle(self) -> None:
        dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        stocks = ["000001.SZ", "600000.SH"]
        market_frames, membership_frame, feature_frames = _synthetic_policy_bundle_parts(dates, stocks=stocks)

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            industry_csv = root / "industry.csv"
            industry_csv.write_text("stock,industry\n000001.SZ,银行\n", encoding="utf-8-sig")
            block_file = root / "infoharbor_block.dat"
            block_file.write_text("#GN_银行,1,880001,20200101,20260519,,\n0#000001\n", encoding="gbk")
            lake = ResearchDataLake(root / "lake")
            first = lake.save_market_data_bundle(
                spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic_a"},
                market_frames=market_frames,
                benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
                membership_frame=membership_frame,
                feature_frames=feature_frames,
                source="synthetic",
            )
            second = lake.save_market_data_bundle(
                spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic_b"},
                market_frames=market_frames,
                benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
                membership_frame=membership_frame,
                feature_frames=feature_frames,
                source="synthetic",
            )
            sector_view = build_sector_board_view_from_policy_bundle(
                lake=lake,
                spec=SectorBoardViewSpec(
                    source_market_dataset_id=first.dataset_id,
                    industry_source_path=str(industry_csv),
                    board_source_path=str(block_file),
                    as_of_date="2026-05-19",
                ),
            )
            with self.assertRaisesRegex(ValueError, "sector_board_view_source_mismatch"):
                load_policy_inputs_from_lake(
                    lake=lake,
                    dataset_id=second.dataset_id,
                    start_date="2026-01-05",
                    end_date="2026-01-07",
                    sector_board_view_id=sector_view.dataset_id,
                )

    def test_policy_input_loader_reports_lake_coverage_blocker_for_missing_benchmark(self) -> None:
        dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        stocks = ["000001.SZ", "000002.SZ"]
        close = pd.DataFrame([[10.0, 20.0], [10.5, 20.5], [11.0, 21.0]], index=dates, columns=stocks)
        market_frames = {
            "Open": close - 0.1,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": pd.DataFrame(1000.0, index=dates, columns=stocks),
            "Amount": pd.DataFrame(10000.0, index=dates, columns=stocks),
        }
        membership_frame = pd.DataFrame(True, index=dates, columns=stocks)

        with TemporaryDirectory() as temp_dir:
            lake = ResearchDataLake(Path(temp_dir))
            record = lake.save_market_data_bundle(
                spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
                market_frames=market_frames,
                benchmark_close=pd.Series(dtype=float, name="000300.SH"),
                membership_frame=membership_frame,
                feature_frames={"score_none": close * 0.0, "score_v2": close * 0.0 + 0.1},
                source="synthetic",
            )
            with self.assertRaisesRegex(ValueError, "lake_coverage_blocker"):
                load_policy_inputs_from_lake(
                    lake=lake,
                    dataset_id=record.dataset_id,
                    start_date="2026-01-05",
                    end_date="2026-01-07",
                )

    def test_lake_loader_allows_single_day_export_preflight_when_requested(self) -> None:
        dates = pd.to_datetime(["2026-01-05"])
        stocks = ["000001.SZ", "000002.SZ"]
        close = pd.DataFrame([[10.0, 20.0]], index=dates, columns=stocks)
        market_frames = {
            "Open": close - 0.1,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": pd.DataFrame(1000.0, index=dates, columns=stocks),
            "Amount": pd.DataFrame(10000.0, index=dates, columns=stocks),
        }
        benchmark_close = pd.Series([4000.0], index=dates, name="000300.SH")
        membership_frame = pd.DataFrame(True, index=dates, columns=stocks)

        with TemporaryDirectory() as temp_dir:
            lake = ResearchDataLake(Path(temp_dir))
            record = lake.save_market_data_bundle(
                spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
                market_frames=market_frames,
                benchmark_close=benchmark_close,
                membership_frame=membership_frame,
                feature_frames={"score_none": close * 0.0, "score_v2": close * 0.0 + 0.1},
                source="synthetic",
            )
            with self.assertRaisesRegex(ValueError, "insufficient_trading_days"):
                load_policy_inputs_from_lake(
                    lake=lake,
                    dataset_id=record.dataset_id,
                    start_date="2026-01-05",
                    end_date="2026-01-05",
                )
            prepared = load_policy_inputs_from_lake(
                lake=lake,
                dataset_id=record.dataset_id,
                start_date="2026-01-05",
                end_date="2026-01-05",
                min_trading_days=1,
            )

        self.assertEqual(len(prepared.close.index), 1)
        self.assertEqual(prepared.raw_cache_meta["lake_coverage_report"]["min_trading_days"], 1)

    def test_policy_input_audit_flags_missing_benchmark_window(self) -> None:
        dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        market_frames, membership_frame, feature_frames = _synthetic_policy_bundle_parts(dates)
        benchmark_close = pd.Series([4000.0, 4020.0], index=dates[[0, 2]], name="000300.SH")

        with TemporaryDirectory() as temp_dir:
            lake = ResearchDataLake(Path(temp_dir))
            record = lake.save_market_data_bundle(
                spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
                market_frames=market_frames,
                benchmark_close=benchmark_close,
                membership_frame=membership_frame,
                feature_frames=feature_frames,
                source="synthetic",
            )
            report = audit_policy_input_bundle(
                lake=lake,
                dataset_id=record.dataset_id,
                windows=[("unit", "2026-01-05", "2026-01-07")],
            )

        self.assertEqual(report["verdict"], "blocked")
        self.assertEqual(report["windows"][0]["verdict"], "blocked")
        self.assertIn("missing_benchmark_close", report["windows"][0]["blockers"])
        self.assertIn("2026-01-06", report["windows"][0]["missing_benchmark_close_dates"])

    def test_policy_input_audit_reports_feature_panel_mismatch(self) -> None:
        dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        market_frames, membership_frame, feature_frames = _synthetic_policy_bundle_parts(dates)
        feature_frames["score_v2"] = feature_frames["score_v2"].iloc[:2].copy()
        benchmark_close = pd.Series([4000.0, 4010.0, 4020.0], index=dates, name="000300.SH")

        with TemporaryDirectory() as temp_dir:
            lake = ResearchDataLake(Path(temp_dir))
            record = lake.save_market_data_bundle(
                spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
                market_frames=market_frames,
                benchmark_close=benchmark_close,
                membership_frame=membership_frame,
                feature_frames=feature_frames,
                source="synthetic",
            )
            report = audit_policy_input_bundle(
                lake=lake,
                dataset_id=record.dataset_id,
                windows=[("unit", "2026-01-05", "2026-01-07")],
            )

        self.assertEqual(report["verdict"], "usable_with_warnings")
        mismatched = {item["feature_name"] for item in report["features"]["mismatched_panels"]}
        self.assertIn("score_v2", mismatched)
        self.assertIn("feature_panel_mismatch", report["warnings"])

    def test_policy_input_audit_sample_nan_scan_reports_feature_nan_ratio(self) -> None:
        dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        market_frames, membership_frame, feature_frames = _synthetic_policy_bundle_parts(dates)
        feature_frames["score_blend"] = feature_frames["score_blend"].copy()
        feature_frames["score_blend"].iloc[1, 0] = float("nan")
        benchmark_close = pd.Series([4000.0, 4010.0, 4020.0], index=dates, name="000300.SH")

        with TemporaryDirectory() as temp_dir:
            lake = ResearchDataLake(Path(temp_dir))
            record = lake.save_market_data_bundle(
                spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
                market_frames=market_frames,
                benchmark_close=benchmark_close,
                membership_frame=membership_frame,
                feature_frames=feature_frames,
                source="synthetic",
            )
            report = audit_policy_input_bundle(
                lake=lake,
                dataset_id=record.dataset_id,
                windows=[("unit", "2026-01-05", "2026-01-07")],
                sample_nan_scan=True,
            )

        self.assertTrue(report["sample_nan_scan"]["enabled"])
        self.assertEqual(report["sample_nan_scan"]["status"], "completed")
        self.assertGreater(report["sample_nan_scan"]["feature_nan_cells"], 0)

    def test_label_completeness_separates_strict_and_realtime_zones(self) -> None:
        trade_dates = pd.to_datetime(
            ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
        )
        sample_frame = pd.DataFrame(
            {
                "date": [
                    "2026-01-05",
                    "2026-01-07",
                    "2026-01-08",
                    "2026-01-09",
                ],
                "stock": ["A", "A", "A", "A"],
            }
        )

        realtime = build_label_completeness_summary(
            sample_frame=sample_frame,
            daily_frame=pd.DataFrame({"date": [dt.strftime("%Y-%m-%d") for dt in trade_dates]}),
            zone="realtime_research",
            available_trade_dates=trade_dates,
            max_forward_horizon=2,
        )
        strict = build_label_completeness_summary(
            sample_frame=sample_frame.loc[sample_frame["date"] <= "2026-01-07"],
            daily_frame=pd.DataFrame({"date": ["2026-01-05", "2026-01-06", "2026-01-07"]}),
            zone="strict_train",
            available_trade_dates=trade_dates,
            max_forward_horizon=2,
        )

        self.assertEqual(realtime["strict_end_date"], "2026-01-07")
        self.assertEqual(realtime["unobserved_label_rows"], 2)
        self.assertLess(realtime["observed_label_row_ratio"], 1.0)
        self.assertEqual(strict["unobserved_label_rows"], 0)
        self.assertEqual(strict["observed_label_row_ratio"], 1.0)

    def test_build_research_database_cli_can_run_market_only_or_skip_market(self) -> None:
        parser = build_research_database.build_parser()
        market_only = parser.parse_args(["--market-only"])
        skip_market = parser.parse_args(["--skip-market", "--zones", "strict_train"])
        defaults = parser.parse_args([])
        option_strings = {option for action in parser._actions for option in action.option_strings}

        self.assertIn("--market-only", option_strings)
        self.assertIn("--skip-market", option_strings)
        self.assertTrue(market_only.market_only)
        self.assertTrue(skip_market.skip_market)
        self.assertEqual(defaults.max_universe_size, 0)

    def test_build_research_database_requires_explicit_lake_dataset_id(self) -> None:
        parser = build_research_database.build_parser()
        defaults = parser.parse_args([])

        self.assertEqual(defaults.data_source, "lake")
        self.assertEqual(defaults.lake_dataset_id, "")
        with self.assertRaisesRegex(ValueError, "explicit --lake-dataset-id"):
            build_research_database.validate_args(defaults)

    def test_import_legacy_training_cache_registers_gold_dataset(self) -> None:
        sample_frame = pd.DataFrame({"date": ["2026-01-05"], "stock": ["000001.SZ"], "action_label": ["open"]})
        daily_frame = pd.DataFrame({"date": ["2026-01-05"], "gross_exposure_target": [0.80]})
        spec = {
            "dataset": "continuous_policy_training_matrices",
            "pool_name": "learned_all_a",
            "benchmark": "000300.SH",
            "start_date": "20260105",
            "end_date": "20260105",
        }

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            save_training_dataset_cache(
                cache_root=root / "legacy",
                spec=spec,
                sample_frame=sample_frame,
                daily_frame=daily_frame,
                teacher_summary={"label_preset": "holdcash_v3"},
            )
            imported = import_legacy_training_dataset_caches(
                cache_root=root / "legacy",
                lake_root=root / "lake",
            )
            lake = ResearchDataLake(root / "lake")
            listed = lake.list_datasets(dataset_kind="continuous_policy_training_matrices")

        self.assertEqual(len(imported), 1)
        self.assertEqual(len(listed), 1)
        self.assertEqual(imported[0]["zone"], "strict_train")

    def test_sharded_training_dataset_registers_and_loads_sorted_frames(self) -> None:
        first_sample = pd.DataFrame(
            {
                "date": ["2026-01-06", "2026-01-05"],
                "stock": ["B", "A"],
                "action_label": ["hold", "open"],
            }
        )
        second_sample = pd.DataFrame(
            {
                "date": ["2026-01-07"],
                "stock": ["A"],
                "action_label": ["reduce"],
            }
        )
        first_daily = pd.DataFrame({"date": ["2026-01-05", "2026-01-06"], "gross_exposure_target": [0.7, 0.8]})
        second_daily = pd.DataFrame({"date": ["2026-01-07"], "gross_exposure_target": [0.6]})
        spec = {
            "dataset": "continuous_policy_training_matrices",
            "pool_name": "learned_all_a",
            "benchmark": "000300.SH",
            "start_date": "2026-01-05",
            "end_date": "2026-01-07",
            "label_preset": "holdcash_v3",
            "sharded": True,
        }

        with TemporaryDirectory() as temp_dir:
            lake = ResearchDataLake(Path(temp_dir))
            identity = lake.build_training_dataset_identity(spec=spec, zone="strict_train")
            root = Path(identity["dataset_dir"])
            sample_dir = root / "sample_frame"
            daily_dir = root / "daily_frame"
            sample_dir.mkdir(parents=True)
            daily_dir.mkdir(parents=True)
            first_sample_path = sample_dir / "2026-01-05_2026-01-06.parquet"
            second_sample_path = sample_dir / "2026-01-07_2026-01-07.parquet"
            first_daily_path = daily_dir / "2026-01-05_2026-01-06.parquet"
            second_daily_path = daily_dir / "2026-01-07_2026-01-07.parquet"
            first_sample.to_parquet(first_sample_path, index=False)
            second_sample.to_parquet(second_sample_path, index=False)
            first_daily.to_parquet(first_daily_path, index=False)
            second_daily.to_parquet(second_daily_path, index=False)
            record = lake.save_sharded_training_dataset(
                spec=spec,
                zone="strict_train",
                shard_records=[
                    {
                        "status": "stored",
                        "start_date": "2026-01-05",
                        "end_date": "2026-01-06",
                        "sample_path": str(first_sample_path),
                        "daily_path": str(first_daily_path),
                        "sample_rows": 2,
                        "daily_rows": 2,
                        "observed_label_rows": 2,
                        "daily_observed_rows": 2,
                    },
                    {
                        "status": "stored",
                        "start_date": "2026-01-07",
                        "end_date": "2026-01-07",
                        "sample_path": str(second_sample_path),
                        "daily_path": str(second_daily_path),
                        "sample_rows": 1,
                        "daily_rows": 1,
                        "observed_label_rows": 1,
                        "daily_observed_rows": 1,
                    },
                ],
                teacher_summary={"label_preset": "holdcash_v3", "sharded": True},
                label_completeness_summary={
                    "zone": "strict_train",
                    "sample_rows": 3,
                    "daily_rows": 3,
                    "observed_label_rows": 3,
                    "unobserved_label_rows": 0,
                    "observed_label_row_ratio": 1.0,
                    "daily_observed_rows": 3,
                    "daily_unobserved_rows": 0,
                    "strict_end_date": "2026-01-07",
                    "is_training_safe": True,
                },
            )
            loaded = lake.load_training_dataset(record.dataset_id)
            described = lake.describe_dataset(record.dataset_id)

        self.assertEqual(record.row_counts["sample_frame"], 3)
        self.assertTrue(described["parameters"]["sharded"])
        self.assertEqual(loaded.sample_frame[["date", "stock"]].values.tolist(), [["2026-01-05", "A"], ["2026-01-06", "B"], ["2026-01-07", "A"]])
        self.assertEqual(loaded.daily_frame["date"].tolist(), ["2026-01-05", "2026-01-06", "2026-01-07"])
        self.assertTrue(loaded.teacher_summary["sharded"])


if __name__ == "__main__":
    unittest.main()
