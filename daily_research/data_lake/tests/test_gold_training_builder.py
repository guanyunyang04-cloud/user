import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from daily_research.continuous_policy.portfolio_simulator import HoldingState, PortfolioState
from daily_research.data_lake import ResearchDataLake
from daily_research.data_lake.audit_gold_dataset import audit_gold_dataset
from daily_research.data_lake.gold_training_builder import (
    GoldBuildSpec,
    build_sharded_gold_training_dataset,
)
from daily_research.data_lake import build_gold_training_dataset


@dataclass
class TinyPrepared:
    universe: tuple[str, ...]
    pool_name: str = "learned_all_a"
    benchmark: str = "000300.SH"
    data_source: str = "synthetic"

    def __post_init__(self) -> None:
        dates = pd.date_range("2026-01-05", periods=8, freq="B")
        self.close = pd.DataFrame(
            {
                "A": [10.0 + idx for idx in range(len(dates))],
                "B": [20.0 + idx for idx in range(len(dates))],
                "C": [30.0 + idx for idx in range(len(dates))],
            },
            index=dates,
        )

    def to_summary(self) -> dict:
        return {
            "pool_name": self.pool_name,
            "benchmark": self.benchmark,
            "data_source": self.data_source,
            "universe_size": len(self.universe),
            "start_date": self.close.index.min().strftime("%Y-%m-%d"),
            "end_date": self.close.index.max().strftime("%Y-%m-%d"),
        }


class TinyFutureMetrics:
    horizons = (1, 2)


class GoldTrainingBuilderTest(unittest.TestCase):
    def test_gold_cli_accepts_iso_dates_for_tq_compatible_request_dates(self) -> None:
        self.assertEqual(build_gold_training_dataset._resolve_start_date("2019-01-02"), "20190102")
        self.assertEqual(build_gold_training_dataset._resolve_end_date("2019-06-30"), "20190630")

    def _spec(self, *, shard_frequency: str = "month") -> GoldBuildSpec:
        return GoldBuildSpec(
            universe="learned_all_a",
            benchmark="000300.SH",
            start_date="2026-01-05",
            end_date="2026-01-14",
            max_universe_size=0,
            data_source="synthetic",
            label_preset="holdcash_v3",
            execution_semantics="semantic_preserving_v1",
            budget_semantics="legacy_total_candidate",
            budget_calibration="none",
            budget_objective="teacher_imitation",
            alpha_prior_source="none",
            random_seed=7,
            skip_multiplier=1.0,
            source_market_dataset_id="policy_input_bundle__synthetic",
            shard_frequency=shard_frequency,
        )

    def _fake_day(self, *, signal_dt, portfolio: PortfolioState, **kwargs):
        date_text = pd.Timestamp(signal_dt).strftime("%Y-%m-%d")
        held_before = float(sum(portfolio.weight_map().values()))
        if not portfolio.holdings:
            actions = [
                {
                    "date": date_text,
                    "stock": "A",
                    "model_action": "open",
                    "weight_change_action": "open",
                    "delta_weight": 0.10,
                }
            ]
            portfolio.holdings["A"] = HoldingState(weight=0.10, entry_price=10.0, peak_price=10.0, hold_days=1)
            portfolio.cash_weight = 0.90
        else:
            actions = [
                {
                    "date": date_text,
                    "stock": "A",
                    "model_action": "hold",
                    "weight_change_action": "hold",
                    "delta_weight": 0.0,
                }
            ]
            for holding in portfolio.holdings.values():
                holding.hold_days += 1
        label_frame = pd.DataFrame(
            {
                "date": [date_text, date_text, date_text],
                "stock": ["A", "B", "C"],
                "action_label": ["hold" if held_before > 0 else "open", "skip", "skip"],
                "current_weight": [held_before, 0.0, 0.0],
                "hold_days": [portfolio.hold_days_map().get("A", 0), 0, 0],
                "teacher_priority": [1.0, 0.0, 0.0],
            }
        )
        daily_row = {"date": date_text, "gross_exposure_target": 0.80}
        return label_frame, daily_row, actions, 0.001, {"budget_objective_is_result_value": 0.0}, {"source_count": 1.0}, {"target_sum": 0.8}

    def test_sharded_gold_builder_preserves_portfolio_state_and_registers_strict_dataset(self) -> None:
        prepared = TinyPrepared(("A", "B", "C"))
        with TemporaryDirectory() as temp_dir, patch(
            "daily_research.data_lake.gold_training_builder._build_day_training_rows",
            side_effect=self._fake_day,
        ):
            lake = ResearchDataLake(Path(temp_dir))
            result = build_sharded_gold_training_dataset(
                lake=lake,
                prepared=prepared,
                future_metrics=TinyFutureMetrics(),
                build_spec=self._spec(shard_frequency="month"),
                zone="strict_train",
                resume=True,
                min_signal_dates=1,
            )
            loaded = lake.load_training_dataset(result["dataset_id"])
            report = audit_gold_dataset(lake=lake, dataset_id=result["dataset_id"])

        self.assertEqual(result["label_completeness_summary"]["unobserved_label_rows"], 0)
        self.assertTrue(result["label_completeness_summary"]["is_training_safe"])
        self.assertEqual(report["status"], "ok")
        self.assertGreaterEqual(int(loaded.sample_frame["current_weight"].max()), 0)
        self.assertGreater(float(loaded.sample_frame["hold_days"].max()), 1.0)

    def test_realtime_gold_marks_tail_labels_unobserved(self) -> None:
        prepared = TinyPrepared(("A", "B", "C"))
        with TemporaryDirectory() as temp_dir, patch(
            "daily_research.data_lake.gold_training_builder._build_day_training_rows",
            side_effect=self._fake_day,
        ):
            lake = ResearchDataLake(Path(temp_dir))
            result = build_sharded_gold_training_dataset(
                lake=lake,
                prepared=prepared,
                future_metrics=TinyFutureMetrics(),
                build_spec=self._spec(shard_frequency="month"),
                zone="realtime_research",
                resume=True,
                min_signal_dates=1,
            )
            loaded = lake.load_training_dataset(result["dataset_id"])
            report = audit_gold_dataset(lake=lake, dataset_id=result["dataset_id"])

        self.assertFalse(result["label_completeness_summary"]["is_training_safe"])
        self.assertGreater(result["label_completeness_summary"]["unobserved_label_rows"], 0)
        self.assertIn("is_observed", loaded.sample_frame.columns)
        self.assertFalse(bool(loaded.sample_frame["is_observed"].all()))
        self.assertEqual(report["status"], "ok")

    def test_resume_skips_completed_shards_without_duplicate_rows(self) -> None:
        prepared = TinyPrepared(("A", "B", "C"))
        with TemporaryDirectory() as temp_dir, patch(
            "daily_research.data_lake.gold_training_builder._build_day_training_rows",
            side_effect=self._fake_day,
        ) as fake_day:
            lake = ResearchDataLake(Path(temp_dir))
            first = build_sharded_gold_training_dataset(
                lake=lake,
                prepared=prepared,
                future_metrics=TinyFutureMetrics(),
                build_spec=self._spec(shard_frequency="month"),
                zone="strict_train",
                resume=True,
                min_signal_dates=1,
            )
            first_calls = fake_day.call_count
            second = build_sharded_gold_training_dataset(
                lake=lake,
                prepared=prepared,
                future_metrics=TinyFutureMetrics(),
                build_spec=self._spec(shard_frequency="month"),
                zone="strict_train",
                resume=True,
                min_signal_dates=1,
            )
            loaded = lake.load_training_dataset(second["dataset_id"])

        self.assertEqual(first["dataset_id"], second["dataset_id"])
        self.assertEqual(fake_day.call_count, first_calls)
        self.assertEqual(int(loaded.sample_frame.duplicated(["date", "stock"]).sum()), 0)

    def test_audit_flags_duplicate_sample_keys(self) -> None:
        spec = {
            "dataset": "continuous_policy_training_matrices",
            "pool_name": "learned_all_a",
            "benchmark": "000300.SH",
            "start_date": "2026-01-05",
            "end_date": "2026-01-05",
            "sharded": True,
        }
        sample = pd.DataFrame({"date": ["2026-01-05", "2026-01-05"], "stock": ["A", "A"], "is_observed": [True, True]})
        daily = pd.DataFrame({"date": ["2026-01-05"], "is_observed": [True]})
        with TemporaryDirectory() as temp_dir:
            lake = ResearchDataLake(Path(temp_dir))
            identity = lake.build_training_dataset_identity(spec=spec, zone="strict_train")
            root = Path(identity["dataset_dir"])
            sample_path = root / "sample_frame" / "dup.parquet"
            daily_path = root / "daily_frame" / "dup.parquet"
            sample_path.parent.mkdir(parents=True)
            daily_path.parent.mkdir(parents=True)
            sample.to_parquet(sample_path, index=False)
            daily.to_parquet(daily_path, index=False)
            record = lake.save_sharded_training_dataset(
                spec=spec,
                zone="strict_train",
                shard_records=[
                    {
                        "status": "stored",
                        "start_date": "2026-01-05",
                        "end_date": "2026-01-05",
                        "sample_path": str(sample_path),
                        "daily_path": str(daily_path),
                        "sample_rows": 2,
                        "daily_rows": 1,
                        "observed_label_rows": 2,
                        "daily_observed_rows": 1,
                    }
                ],
                teacher_summary={},
                label_completeness_summary={
                    "zone": "strict_train",
                    "sample_rows": 2,
                    "daily_rows": 1,
                    "observed_label_rows": 2,
                    "unobserved_label_rows": 0,
                    "daily_observed_rows": 1,
                    "daily_unobserved_rows": 0,
                    "observed_label_row_ratio": 1.0,
                    "strict_end_date": "2026-01-05",
                    "is_training_safe": True,
                },
            )
            report = audit_gold_dataset(lake=lake, dataset_id=record.dataset_id)

        self.assertEqual(report["status"], "failed")
        self.assertIn("duplicate_sample_keys", {item["code"] for item in report["findings"]})


if __name__ == "__main__":
    unittest.main()
