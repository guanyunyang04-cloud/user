import unittest

import numpy as np
import pandas as pd

from daily_research.continuous_policy.model_portfolio_set_v5 import (
    PortfolioSetDayDataset,
    build_portfolio_set_v5_targets,
    collate_portfolio_set_days,
)


class PortfolioSetV5DaySetDatasetTest(unittest.TestCase):
    def _sample_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": ["2026-01-05", "2026-01-05", "2026-01-06"],
                "stock": ["SRC", "RCV", "KEEP"],
                "alpha_score": [0.1, 0.9, 0.2],
                "current_weight": [0.20, 0.0, 0.18],
                "action_label": ["reduce", "open", "hold"],
                "portfolio_daily_target_delta_intent": [-0.06, 0.08, 0.0],
                "portfolio_daily_source_release_preference": [0.90, 0.0, 0.0],
                "portfolio_daily_source_forward_proxy_keep_risk": [0.05, 0.0, 0.95],
                "portfolio_daily_source_economic_block_risk": [0.02, 0.0, 0.02],
                "portfolio_daily_receiver_score": [0.0, 0.92, 0.0],
                "portfolio_daily_unified_receiver_score": [0.0, 0.90, 0.0],
            }
        )

    def test_targets_create_source_receiver_and_masks(self) -> None:
        sample = self._sample_frame()

        targets = build_portfolio_set_v5_targets(sample)

        self.assertGreater(float(targets.loc[0, "source_supply_score"]), 0.0)
        self.assertGreater(float(targets.loc[1, "receiver_demand_score"]), 0.0)
        self.assertEqual(float(targets.loc[2, "source_supply_score"]), 0.0)
        self.assertEqual(float(targets.loc[0, "source_mask"]), 1.0)
        self.assertEqual(float(targets.loc[1, "receiver_mask"]), 1.0)

    def test_day_set_collator_builds_masked_batches_without_full_attention_state(self) -> None:
        sample = self._sample_frame()
        daily = pd.DataFrame(
            {
                "date": ["2026-01-05", "2026-01-06"],
                "market_downside_pressure": [0.1, 0.2],
                "gross_exposure_target": [0.7, 0.6],
            }
        )
        targets = build_portfolio_set_v5_targets(sample)
        dataset = PortfolioSetDayDataset(
            sample_frame=sample,
            daily_frame=daily,
            static_feature_names=["alpha_score", "current_weight"],
            sequence_bases=[],
            daily_feature_names=["market_downside_pressure"],
            targets=targets,
            static_fill=np.zeros(2, dtype=np.float32),
            static_means=np.zeros(2, dtype=np.float32),
            static_stds=np.ones(2, dtype=np.float32),
            sequence_fill=np.zeros(1, dtype=np.float32),
            sequence_means=np.zeros(1, dtype=np.float32),
            sequence_stds=np.ones(1, dtype=np.float32),
            daily_fill=np.zeros(1, dtype=np.float32),
            daily_means=np.zeros(1, dtype=np.float32),
            daily_stds=np.ones(1, dtype=np.float32),
        )

        batch = collate_portfolio_set_days([dataset[0], dataset[1]])

        self.assertEqual(tuple(batch["static_x"].shape), (2, 2, 2))
        self.assertEqual(tuple(batch["sequence_x"].shape), (2, 2, 1, 1))
        self.assertEqual(tuple(batch["sample_mask"].shape), (2, 2))
        self.assertEqual(int(batch["sample_mask"][0].sum().item()), 2)
        self.assertEqual(int(batch["sample_mask"][1].sum().item()), 1)
        self.assertGreater(float(batch["source_mask"].sum().item()), 0.0)
        self.assertGreater(float(batch["receiver_mask"].sum().item()), 0.0)

    def test_missing_date_column_fails_safely(self) -> None:
        with self.assertRaises(ValueError):
            build_portfolio_set_v5_targets(self._sample_frame().drop(columns=["date"]))


if __name__ == "__main__":
    unittest.main()
