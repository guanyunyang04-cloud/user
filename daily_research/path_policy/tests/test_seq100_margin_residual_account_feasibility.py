from __future__ import annotations

import numpy as np
import pandas as pd

from daily_research.path_policy import (
    seq100_margin_residual_account_feasibility as feasibility,
)


def test_account_contract_is_single_configuration_and_nonpromotional() -> None:
    study, _ = feasibility.load_study()

    assert study["account"]["starting_cash_cny"] == 1_000_000.0
    assert study["account"]["daily_ranked_candidate_count"] == 48
    assert study["account"]["position_slots"] == 48
    assert study["account"]["cost_scenario"] == "double_slippage"
    assert study["interpretation"]["source_confirmation_gate_passed"] is False
    assert study["interpretation"][
        "account_success_cannot_override_confirmation_failure"
    ]
    assert study["decision_boundary"]["profit_claim_allowed"] is False
    assert study["source"]["adjust_factor_dataset_id"].startswith("adjust_factor__")
    assert "total-return equivalent" in study["account"]["corporate_action_accounting"]


def test_account_source_preserves_passed_development_and_failed_confirmation() -> None:
    study, _ = feasibility.load_study()
    source = feasibility._source_contract(study)

    assert source["development"]["gate"]["passed"] is True
    assert source["confirmation"]["gate"]["passed"] is False
    for year in range(2014, 2026):
        assert feasibility._forecast_path_for_year(source, year).is_file()


def test_risk_budget_uses_only_the_declared_development_boundary() -> None:
    dates = pd.bdate_range("2021-01-04", periods=520)
    daily = np.where(np.arange(len(dates)) % 2 == 0, 0.01, -0.009)
    equity = 1_000_000.0 * np.cumprod(1.0 + daily)
    frame = pd.DataFrame(
        {
            "trade_date": dates.strftime("%Y-%m-%d"),
            "equity": equity,
            "inside_signal_period": True,
        }
    )
    config = {
        "derivation_end_date": "2022-12-31",
        "target_annualized_volatility": 0.15,
        "minimum_gross_fraction": 0.25,
        "maximum_gross_fraction": 1.0,
    }

    result = feasibility._derive_risk_budget(
        frame,
        starting_cash=1_000_000.0,
        config=config,
    )

    assert result["confirmation_data_used_in_derivation"] is False
    assert result["parameter_grid_performed"] is False
    assert 0.25 <= result["frozen_target_gross_fraction"] <= 1.0
