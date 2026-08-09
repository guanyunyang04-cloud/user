from __future__ import annotations

from daily_research.path_policy import (
    seq100_exact_value_growth_top3_risk_frontier as frontier,
)


def test_load_study_freezes_round_number_grid() -> None:
    study, _ = frontier.load_study()
    assert study["gross_fraction_grid"] == [0.4, 0.45, 0.5, 0.55]
    assert [row["horizon"] for row in study["policies"]] == [60, 120]
    assert study["account_contract"]["slots"] == 3


def test_choice_uses_best_growth_and_lower_exposure_only_for_near_tie() -> None:
    rows = [
        {
            "policy": "fixed_d60_max3",
            "gross_fraction": 0.45,
            "annualized_log_growth": 0.090,
            "annualized_compound_return": 0.094,
            "annualized_volatility": 0.13,
            "maximum_drawdown": -0.22,
            "positive_year_count": 10,
            "gate_passed": True,
        },
        {
            "policy": "fixed_d60_max3",
            "gross_fraction": 0.50,
            "annualized_log_growth": 0.092,
            "annualized_compound_return": 0.096,
            "annualized_volatility": 0.14,
            "maximum_drawdown": -0.24,
            "positive_year_count": 10,
            "gate_passed": True,
        },
        {
            "policy": "fixed_d120_max3",
            "gross_fraction": 0.55,
            "annualized_log_growth": 0.11,
            "annualized_compound_return": 0.116,
            "annualized_volatility": 0.16,
            "maximum_drawdown": -0.31,
            "positive_year_count": 11,
            "gate_passed": False,
        },
    ]
    decision = frontier._choice(rows, near_tie_tolerance=0.0025)
    assert decision["selected_policy"] == "fixed_d60_max3"
    assert decision["selected_gross_fraction"] == 0.45
    assert decision["passing_grid_point_count"] == 2
