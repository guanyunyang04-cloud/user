from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_v4_economic_realizability as old_economic
from daily_research.path_policy.seq100_candidate_execution import (
    ExecutionCosts as OldExecutionCosts,
)
from daily_research.path_policy.seq100_full_market_multitask_forecast import (
    _simulate_account_spec as old_simulate_account,
)
from daily_research.technical.portfolio import (
    ExecutionCosts,
    buy_position,
    sell_position,
    simulate_account,
)


def _costs() -> ExecutionCosts:
    return ExecutionCosts(
        lot_size=100,
        commission_bps=3.0,
        minimum_commission_cny=5.0,
        transfer_fee_bps=0.1,
        slippage_bps=7.0,
        stress_slippage_multiplier=2.0,
        stamp_tax_schedule=(("1900-01-01", 10.0), ("2023-08-28", 5.0)),
    )


def _old_costs() -> OldExecutionCosts:
    current = _costs()
    return OldExecutionCosts(**current.__dict__)


def test_buy_and_sell_cashflows_match_legacy_implementation() -> None:
    arguments = {
        "available_cash": 100_000.0,
        "allocated_cash": 37_500.0,
        "symbol_idx": 3,
        "signal_date_idx": 20,
        "execution_date_idx": 21,
        "raw_open": 12.34,
        "adjusted_open": 10.0,
        "slippage_multiplier": 2.0,
    }
    current_position, current_buy = buy_position(**arguments, costs=_costs())
    old_position, old_buy = old_economic._buy_position(**arguments, costs=_old_costs())
    assert current_position is not None and old_position is not None
    assert current_position.__dict__ == old_position.__dict__
    assert current_buy == old_buy
    current_proceeds, current_sell = sell_position(
        position=current_position,
        adjusted_open=10.75,
        trade_date="2024-01-05",
        costs=_costs(),
        slippage_multiplier=2.0,
    )
    old_proceeds, old_sell = old_economic._sell_position(
        position=old_position,
        adjusted_open=10.75,
        trade_date="2024-01-05",
        costs=_old_costs(),
        slippage_multiplier=2.0,
    )
    assert current_proceeds == old_proceeds
    assert current_sell == old_sell


def test_account_replay_matches_legacy_engine() -> None:
    date_values = np.asarray(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    )
    symbol_values = ["000001.SZ", "600000.SH"]
    daily_raw = np.full((5, 2, 13), np.nan, dtype=np.float32)
    daily_raw[:, :, 0] = np.asarray([[10.0, 20.0]] * 5)
    daily_raw[:, :, 3] = np.asarray(
        [[10.0, 20.0], [10.1, 20.1], [10.2, 20.2], [10.3, 20.3], [10.4, 20.4]]
    )
    raw_open = np.asarray([[10.0, 20.0]] * 5, dtype=np.float32)
    entry_filled = np.ones((5, 2), dtype=bool)
    selections = pd.DataFrame(
        {
            "candidate_id": [1, 2],
            "date_idx": [0, 1],
            "trade_date": date_values[:2],
            "symbol": symbol_values,
            "symbol_idx": [0, 1],
            "selection_rank": [1, 1],
            "legal_gross_return": [0.05, -0.03],
            "fill_day": [2, 2],
            "fold": [1, 1],
            "variant": ["test", "test"],
            "exit_policy": ["planned_close", "planned_close"],
            "gate": ["always", "always"],
            "top_k": [1, 1],
            "cost_scenario": ["base", "base"],
            "take_profit_hit": [False, False],
        }
    )
    spec = {
        "variant": "test",
        "exit_policy": "planned_close",
        "gate": "always",
        "top_k": 1,
        "cost_scenario": "base",
        "slippage_multiplier": 1.0,
        "cohort_equity_fraction": 0.5,
        "planned_fill_day": 2,
    }
    current_result, current_equity, current_trades = simulate_account(
        spec=spec,
        selections=selections,
        date_values=date_values,
        symbol_values=symbol_values,
        cutoff_idx=4,
        daily_raw=daily_raw,
        raw_open=raw_open,
        entry_filled=entry_filled,
        costs=_costs(),
    )
    context = SimpleNamespace(
        pack={"symbol_values": symbol_values},
        cutoff_idx=4,
        date_values=date_values,
    )
    old_result, old_equity, old_trades = old_simulate_account(
        spec=spec,
        selections=selections,
        context=context,
        daily_raw=daily_raw,
        raw_open=raw_open,
        entry_filled=entry_filled,
        costs=_old_costs(),
    )
    comparable = {
        "starting_cash",
        "ending_equity",
        "total_net_return",
        "maximum_drawdown",
        "trade_count",
        "minimum_cash",
        "maximum_position_count",
        "maximum_unique_symbol_count",
        "maximum_same_symbol_open_cohorts",
        "same_symbol_overlap_order_count",
        "same_symbol_overlap_filled_count",
        "same_symbol_overlap_skipped_count",
    }
    for key in comparable:
        assert current_result[key] == old_result[key]
    pd.testing.assert_frame_equal(current_equity, old_equity)
    pd.testing.assert_frame_equal(current_trades, old_trades)
