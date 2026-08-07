from __future__ import annotations

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_daily_phase_execution_validate as validate


def test_stamp_tax_schedule_changes_on_effective_date() -> None:
    result = validate._stamp_tax_bps(
        np.asarray(["2023-08-27", "2023-08-28"]),
        (("1900-01-01", 10.0), ("2023-08-28", 5.0)),
    )
    assert np.array_equal(result, np.asarray([10.0, 5.0]))


def test_cash_and_position_paths_reconstruct_entry_and_exit() -> None:
    equity = pd.DataFrame({"date_idx": [10, 11, 12, 13]})
    trades = pd.DataFrame(
        {
            "entry_date_idx": [11],
            "exit_date_idx": [13],
            "buy_cash_cny": [400.0],
            "sell_proceeds_cny": [440.0],
        }
    )
    cash, positions = validate._cash_and_position_paths(
        equity, trades, starting_cash=1_000.0
    )
    assert np.array_equal(cash, np.asarray([1000.0, 600.0, 600.0, 1040.0]))
    assert np.array_equal(positions, np.asarray([0, 1, 1, 0]))


def test_hac_mean_preserves_constant_series() -> None:
    result = validate._hac_mean(np.full(30, 0.02), lag=5)
    assert np.isclose(result["mean"], 0.02)
    assert result["se"] < 1.0e-12


def test_newey_west_regression_recovers_linear_relation() -> None:
    market = np.linspace(-0.02, 0.02, 200)
    outcome = 0.001 + 0.75 * market
    result = validate._newey_west_regression(outcome, market, lag=5)
    assert abs(result["daily_alpha"] - 0.001) < 1.0e-12
    assert abs(result["market_beta"] - 0.75) < 1.0e-12
