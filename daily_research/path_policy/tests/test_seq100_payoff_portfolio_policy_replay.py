from __future__ import annotations

import pandas as pd
import pytest

from daily_research.path_policy.seq100_payoff_portfolio_policy_replay import (
    HORIZONS,
    PortfolioPolicyReplayError,
    build_policy_specs,
    common_model_rows,
)


def test_build_policy_specs_freezes_only_overlap_and_cap_diagnostics() -> None:
    specs = build_policy_specs(
        horizon=5,
        variant="frozen_score_d5",
        cohort_equity_fraction=0.2,
    )

    assert len(specs) == 4
    assert {bool(spec["allow_overlapping_same_symbol"]) for spec in specs} == {
        False,
        True,
    }
    assert {spec.get("maximum_credited_gross_return") for spec in specs} == {
        None,
        0.10,
    }
    assert all(spec["top_k"] == 10 for spec in specs)
    assert all(spec["cost_scenario"] == "stress" for spec in specs)
    assert all(spec["planned_fill_day"] == 5 for spec in specs)
    assert all(spec["cohort_equity_fraction"] == pytest.approx(0.2) for spec in specs)


def test_common_model_rows_intersects_all_horizons() -> None:
    schedules = {
        horizon: pd.DataFrame(
            {
                "model_row_position": [1, 2, 3] if horizon != 10 else [2, 3, 4],
                "top_k": [10, 10, 10],
            }
        )
        for horizon in HORIZONS
    }
    assert common_model_rows(schedules) == {2, 3}


def test_common_model_rows_rejects_duplicate_schedule_rows() -> None:
    schedules = {
        horizon: pd.DataFrame({"model_row_position": [1, 2], "top_k": [10, 10]})
        for horizon in HORIZONS
    }
    schedules[3] = pd.DataFrame({"model_row_position": [1, 1], "top_k": [10, 10]})
    with pytest.raises(PortfolioPolicyReplayError, match="duplicate model row"):
        common_model_rows(schedules)
