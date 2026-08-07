from __future__ import annotations

import functools
import math

import numpy as np

from daily_research.path_policy import seq100_dynamic_oracle as oracle
from daily_research.path_policy import seq100_dynamic_oracle_validate as validate
from daily_research.path_policy import seq100_market_replay as replay
from daily_research.path_policy.seq100_candidate_execution import ExecutionCosts


def _market(*, capacity_amount: float = 1_000_000_000.0) -> replay.ReplayMarket:
    dates = np.asarray(
        [
            "2023-01-03",
            "2023-01-04",
            "2023-01-05",
            "2023-01-06",
            "2023-01-09",
            "2023-01-10",
        ],
        dtype=object,
    )
    close = np.asarray(
        [
            [10.0, 10.0],
            [11.0, 9.0],
            [13.0, 8.0],
            [12.0, 14.0],
            [15.0, 13.0],
            [16.0, 17.0],
        ],
        dtype=np.float32,
    )
    entry_open = np.asarray(
        [
            [10.0, 10.0],
            [10.0, 10.0],
            [11.0, 9.0],
            [13.0, 8.0],
            [12.0, 14.0],
            [15.0, 13.0],
        ],
        dtype=np.float32,
    )
    costs = ExecutionCosts(
        lot_size=100,
        commission_bps=3.0,
        minimum_commission_cny=5.0,
        transfer_fee_bps=0.1,
        slippage_bps=7.0,
        stress_slippage_multiplier=2.0,
        stamp_tax_schedule=(("1900-01-01", 10.0),),
    )
    return replay.ReplayMarket(
        date_values=dates,
        symbol_values=np.asarray(["A", "B"], dtype=object),
        start_idx=0,
        end_idx=5,
        entry_open_raw=entry_open,
        exit_close_raw=close,
        exit_sellable=np.ones_like(close, dtype=bool),
        entry_filled=np.ones_like(close, dtype=bool),
        amount_panel=np.full_like(close, capacity_amount, dtype=np.float32),
        quality_mask=np.ones_like(close, dtype=bool),
        mark_close=close.copy(),
        costs=costs,
        quality_rows=int(close.size),
    )


def _brute_cash_log(market: replay.ReplayMarket, cost_scenario: str) -> float:
    buy_multiplier, sell_multiplier = replay.proportional_cost_multipliers(
        market, cost_scenario
    )

    @functools.cache
    def best(signal_idx: int) -> float:
        if signal_idx >= market.end_idx:
            return 0.0
        result = best(signal_idx + 1)
        for symbol_idx in range(market.symbol_count):
            if not bool(market.quality_mask[signal_idx, symbol_idx]):
                continue
            if not bool(market.entry_filled[signal_idx, symbol_idx]):
                continue
            entry_idx = signal_idx + 1
            entry = float(market.entry_open_raw[entry_idx, symbol_idx])
            for exit_idx in range(signal_idx + 2, market.end_idx + 1):
                if not bool(market.exit_sellable[exit_idx, symbol_idx]):
                    continue
                exit_price = float(market.exit_close_raw[exit_idx, symbol_idx])
                factor = (
                    exit_price
                    * float(sell_multiplier[exit_idx - market.start_idx])
                    / (entry * buy_multiplier)
                )
                result = max(result, math.log(factor) + best(exit_idx))
        return result

    return best(market.start_idx)


def test_relaxed_bellman_matches_exhaustive_variable_duration_search() -> None:
    market = _market()
    study = oracle.load_study()
    for cost_scenario in replay.COST_SCENARIOS:
        solution = oracle.solve_relaxed_oracle(
            market=market,
            study=study,
            cost_scenario=cost_scenario,
            output_root=oracle.DEFAULT_OUTPUT_ROOT,
            write_outputs=False,
        )
        expected = _brute_cash_log(market, cost_scenario)
        assert math.isclose(
            solution.terminal_log_wealth, expected, abs_tol=1.0e-12, rel_tol=0.0
        )
        assert math.isclose(
            solution.selected_trades["proportional_log_return"].sum(),
            expected,
            abs_tol=1.0e-12,
            rel_tol=0.0,
        )
        forward = validate.independent_forward_ceiling(market, cost_scenario)
        assert math.isclose(forward[-1], expected, abs_tol=1.0e-12, rel_tol=0.0)


def test_quality_membership_and_t1_are_hard_constraints() -> None:
    market = _market()
    market.quality_mask[:, 1] = False
    solution = oracle.solve_relaxed_oracle(
        market=market,
        study=oracle.load_study(),
        cost_scenario="base",
        output_root=oracle.DEFAULT_OUTPUT_ROOT,
        write_outputs=False,
    )
    assert set(solution.selected_trades["symbol_idx"].astype(int)) <= {0}
    assert bool(
        (
            solution.selected_trades["exit_date_idx"].astype(int)
            >= solution.selected_trades["signal_date_idx"].astype(int) + 2
        ).all()
    )


def test_finite_replay_is_cash_conserving_and_capacity_bounded() -> None:
    market = _market(capacity_amount=1_000_000.0)
    solution = oracle.solve_relaxed_oracle(
        market=market,
        study=oracle.load_study(),
        cost_scenario="base",
        output_root=oracle.DEFAULT_OUTPUT_ROOT,
        write_outputs=False,
    )
    uncapped, _, uncapped_trades = oracle.replay_finite_account(
        market=market,
        selected_trades=solution.selected_trades,
        cost_scenario="base",
        starting_cash=1_000_000.0,
        capacity_mode="uncapped",
        maximum_signal_amount_fraction=0.005,
    )
    capped, capped_equity, capped_trades = oracle.replay_finite_account(
        market=market,
        selected_trades=solution.selected_trades,
        cost_scenario="base",
        starting_cash=1_000_000.0,
        capacity_mode="signal_amount_0p5pct",
        maximum_signal_amount_fraction=0.005,
    )
    assert uncapped["ending_equity_cny"] > 0.0
    assert capped["ending_equity_cny"] > 0.0
    assert bool((capped_equity["cash"].astype(float) >= -1.0e-8).all())
    assert bool((capped_equity["position_count"].astype(int) <= 1).all())
    if not capped_trades.empty:
        assert float(capped_trades["signal_amount_participation"].max()) <= 0.005 + 1.0e-12
    assert len(uncapped_trades) <= len(solution.selected_trades)
