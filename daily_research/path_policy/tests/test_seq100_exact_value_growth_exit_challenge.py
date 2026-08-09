from __future__ import annotations

import pytest

from daily_research.path_policy import seq100_exact_value_growth_exit_challenge as study


def test_extended_fixed_policy_allows_d120_only_inside_isolated_challenge() -> None:
    study.ExtendedFixedPolicy(name="fixed_d120", kind="fixed", fixed_day=120).validate()
    with pytest.raises(ValueError, match="invalid extended fixed policy"):
        study.ExtendedFixedPolicy(
            name="fixed_d121", kind="fixed", fixed_day=121
        ).validate()


def test_account_comparison_requires_risk_control() -> None:
    config, _ = study.load_study()
    baseline = {
        "liquidated_ending_equity_cny": 2_800_000.0,
        "annualized_log_growth": 0.08,
        "signal_period_maximum_drawdown": -0.29,
        "signal_period_annualized_volatility": 0.14,
        "positive_year_count": 11,
    }
    challenger = {
        "liquidated_ending_equity_cny": 3_100_000.0,
        "annualized_log_growth": 0.088,
        "signal_period_maximum_drawdown": -0.31,
        "signal_period_annualized_volatility": 0.151,
        "positive_year_count": 11,
    }
    comparisons, decision = study._account_comparison(
        {
            "risk_budget": {60: baseline, 120: challenger},
            "full": {60: baseline, 120: challenger},
        },
        config,
    )
    assert len(comparisons) == 2
    assert decision["adaptive_retrospective_d120_challenger_passed"]
    assert decision["not_an_untouched_confirmation"]
