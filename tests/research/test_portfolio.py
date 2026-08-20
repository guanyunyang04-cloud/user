from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.research.portfolio import (
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


def test_buy_and_sell_cashflows_respect_lots_and_costs() -> None:
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
    position, buy = buy_position(**arguments, costs=_costs())
    assert position is not None
    assert position.shares == 3000
    assert buy["total_cost"] > 0.0
    proceeds, sell = sell_position(
        position=position,
        adjusted_open=10.75,
        trade_date="2024-01-05",
        costs=_costs(),
        slippage_multiplier=2.0,
    )
    assert proceeds > 0.0
    assert sell["stamp_tax"] > 0.0
    assert sell["total_cost"] > 0.0


def test_account_replay_is_finite_and_respects_t1() -> None:
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
    result, equity, trades = simulate_account(
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
    assert result["forbidden_2026_read_count"] == 0
    assert np.isfinite(result["ending_equity"])
    assert result["trade_count"] == 2
    assert result["maximum_position_count"] == 1
    assert len(equity) == len(date_values)
    assert len(trades) == 2
    assert (trades["entry_date_idx"] > trades["signal_date_idx"]).all()
    assert (trades["exit_date_idx"] >= trades["entry_date_idx"]).all()
