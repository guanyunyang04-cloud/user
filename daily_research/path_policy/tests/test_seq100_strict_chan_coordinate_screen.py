from __future__ import annotations

import pandas as pd
import pytest

from daily_research.path_policy import seq100_strict_chan_coordinate_screen as screen


def test_low_direction_and_nested_composites_are_not_double_reversed() -> None:
    study = screen._load_study()
    values = {}
    for name, direction in study["coordinates"].items():
        values[name] = 0.1 if direction == "low" else 0.9
    scored, eligible = screen._add_scores(pd.DataFrame([values]), study)
    assert scored.iloc[0]["score__minute_last_30m_return_rank"] == pytest.approx(0.9)
    assert scored.iloc[0]["score__anti_chase"] == pytest.approx(0.9)
    assert scored.iloc[0]["score__quality_flow"] == pytest.approx(0.9)
    assert scored.iloc[0]["score__supportive_regime"] == pytest.approx(0.9)
    assert scored.iloc[0]["score__combined"] == pytest.approx(0.9)
    assert eligible["market_volatility_rank"] is False
    assert eligible["combined"] is True


def test_coordinate_gate_requires_both_time_splits() -> None:
    study = screen._load_study()
    common = {
        "coordinate": "combined",
        "favorable_event_count": 40,
        "favorable_case_count": 30,
        "favorable_case_mean_net_base": 0.02,
        "favorable_case_mean_net_stress": 0.01,
        "favorable_minus_unfavorable_base": 0.01,
        "bootstrap_lift_low": 0.001,
    }
    contrasts = pd.DataFrame(
        [
            {"evaluation_split": "development", **common},
            {"evaluation_split": "validation", **common},
        ]
    )
    result = screen._gate(
        contrasts,
        study=study,
        gate_eligible={"combined": True},
    )
    assert bool(result.iloc[0]["strict_gate_passed"])

    failed = screen._gate(
        contrasts.iloc[:1],
        study=study,
        gate_eligible={"combined": True},
    )
    assert not bool(failed.iloc[0]["strict_gate_passed"])
    assert "validation:missing" in failed.iloc[0]["failure_reasons"]
