from __future__ import annotations

import pandas as pd

from daily_research.path_policy.rl_dataset import build_path20_sequence_trajectory_dataset
from daily_research.path_policy.rl_replay import run_sequence_policy_replay
from daily_research.path_policy.tests.fixtures import make_prepared_policy_inputs


def test_sequence_replay_writes_projection_diagnostics_and_returns() -> None:
    prepared = make_prepared_policy_inputs(days=35, stocks=("AAA", "BBB", "CCC"))
    trajectory = build_path20_sequence_trajectory_dataset(
        prepared,
        start_date="20240102",
        end_date="20240209",
        lake_dataset_id="policy_input_bundle__fixture",
        sequence_length=4,
        min_trading_days=2,
    )
    target_by_date = {
        dt.strftime("%Y-%m-%d"): pd.Series({"AAA": 0.25, "BBB": 0.10, "CCC": 0.0})
        for dt in trajectory.dates
    }

    result = run_sequence_policy_replay(
        trajectory=trajectory,
        target_weight_by_date=target_by_date,
        execution_price_frame=prepared.open_.shift(-1),
        max_position_weight=0.20,
        max_gross_exposure=0.50,
        max_positions=2,
        turnover_budget=1.0,
    )

    assert result["metrics"]["return_count"] > 0
    assert not result["projection_diagnostics"].empty
    assert result["projection_diagnostics"]["projected_gross_exposure"].max() <= 0.50 + 1.0e-6
    assert "projection_l1_distance" in result["projection_diagnostics"].columns
