from __future__ import annotations

from daily_research.path_policy import seq100_exact_value_growth_account as account


def test_contract_freezes_natural_account_and_development_only_risk_budget() -> None:
    study, _ = account.load_study()

    assert study["selection"]["monthly_order_count"] == 10
    assert study["selection"]["candidate_scan_count"] == 30
    assert study["account"]["position_slots"] == 30
    assert study["account"]["cost_scenario"] == "double_slippage"
    assert study["account"]["pyramiding"] is False
    assert study["risk_budget"]["derivation_end_date"] == "2019-12-31"
    assert study["risk_budget"]["parameter_grid_performed"] is False


def test_source_information_gate_is_required_and_currently_passes() -> None:
    study, _ = account.load_study()
    sources = account._source_contract(study)

    assert sources["exact_policy_summary"].is_file()
    assert sources["candidate_features"].is_file()


def test_account_gate_rejects_excess_drawdown_even_when_profitable() -> None:
    metric = {
        "liquidated_total_return": 1.0,
        "signal_period_cagr_trading_days": 0.1,
        "signal_period_maximum_drawdown": -0.31,
        "signal_period_annualized_volatility": 0.15,
        "ruined": False,
    }
    annual = [{"net_return": 0.01} for _ in range(14)]
    contract = {
        "minimum_positive_years": 10,
        "maximum_drawdown": 0.30,
        "maximum_annualized_volatility_for_risk_account": 0.20,
    }

    result = account._account_gate(metric, annual, contract=contract, risk_account=True)

    assert result["checks"]["maximum_drawdown_within_limit"] is False
    assert result["passed"] is False


def test_account_gate_applies_risk_volatility_limit_only_to_risk_account() -> None:
    metric = {
        "liquidated_total_return": 1.0,
        "signal_period_cagr_trading_days": 0.1,
        "signal_period_maximum_drawdown": -0.20,
        "signal_period_annualized_volatility": 0.25,
        "ruined": False,
    }
    annual = [{"net_return": 0.01} for _ in range(14)]
    contract = {
        "minimum_positive_years": 10,
        "maximum_drawdown": 0.30,
        "maximum_annualized_volatility_for_risk_account": 0.20,
    }

    full = account._account_gate(metric, annual, contract=contract, risk_account=False)
    risk = account._account_gate(metric, annual, contract=contract, risk_account=True)

    assert full["passed"] is True
    assert risk["checks"]["annualized_volatility_within_limit"] is False
    assert risk["passed"] is False
