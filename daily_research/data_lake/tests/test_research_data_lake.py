import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from daily_research.data_lake import ResearchDataLake, build_label_completeness_summary
from daily_research.data_lake import build_research_database
from daily_research.data_lake.import_legacy_training_caches import import_legacy_training_dataset_caches
from daily_research.continuous_policy.training_dataset_cache import save_training_dataset_cache


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
        option_strings = {option for action in parser._actions for option in action.option_strings}

        self.assertIn("--market-only", option_strings)
        self.assertIn("--skip-market", option_strings)
        self.assertTrue(market_only.market_only)
        self.assertTrue(skip_market.skip_market)

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


if __name__ == "__main__":
    unittest.main()
