from __future__ import annotations

from daily_research.path_policy.oracle import run_oracle_path20_rollout
from daily_research.path_policy.tests.fixtures import make_prepared_policy_inputs


def test_oracle_path20_rollout_completes_with_target_weight_execution() -> None:
    prepared = make_prepared_policy_inputs(days=45, stocks=("AAA", "BBB", "CCC", "DDD", "EEE"))

    result = run_oracle_path20_rollout(
        prepared=prepared,
        start_date="20240102",
        end_date="20240131",
        execution_mode="next_open",
        max_position_weight=0.20,
        max_gross_exposure=0.80,
        max_positions=3,
        turnover_budget=1.0,
    )

    assert result["metrics"]["return_count"] > 0
    assert not result["action_panel"].empty
    assert not result["turnover_frame"].empty
    assert not result["oracle_daily"].empty
    assert "native_target_valid_rate" in result["metrics"]
    assert result["turnover_frame"]["path_policy_projected_target_weight_sum"].max() <= 0.80 + 1.0e-6
