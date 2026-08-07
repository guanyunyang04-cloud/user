from __future__ import annotations

import functools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_dynamic_oracle as oracle
from daily_research.path_policy import seq100_dynamic_oracle_resolution as resolution
from daily_research.path_policy import seq100_market_replay as replay
from daily_research.path_policy.seq100_candidate_execution import ExecutionCosts


def _market() -> replay.ReplayMarket:
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
        amount_panel=np.full_like(close, 1_000_000.0, dtype=np.float32),
        quality_mask=np.ones_like(close, dtype=bool),
        mark_close=close.copy(),
        costs=costs,
        quality_rows=int(close.size),
    )


def _file_record(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": replay.sha256(path)}


def _write_solution_arrays(
    tmp_path: Path, solution: oracle.RelaxedSolution
) -> dict[str, object]:
    root = tmp_path / "relaxed_oracle" / f"cost={solution.cost_scenario}"
    return {
        "cash_action_symbol": _file_record(root / "cash_action_symbol.npy"),
        "cash_log_value": _file_record(root / "cash_log_value.npy"),
        "holding_exit_policy": _file_record(root / "holding_exit_policy.npy"),
    }


def _brute_prefix_advantage(
    market: replay.ReplayMarket,
    *,
    signal_idx: int,
    symbol_idx: int,
    terminal_idx: int,
    cost_scenario: str,
) -> float:
    buy_multiplier, sell_multiplier = replay.proportional_cost_multipliers(
        market, cost_scenario
    )

    @functools.cache
    def cash_value(state_idx: int) -> float:
        if state_idx >= terminal_idx:
            return 0.0
        result = cash_value(state_idx + 1)
        for candidate in range(market.symbol_count):
            if not bool(market.quality_mask[state_idx, candidate]):
                continue
            if not bool(market.entry_filled[state_idx, candidate]):
                continue
            entry_idx = state_idx + 1
            entry = float(market.entry_open_raw[entry_idx, candidate])
            for exit_idx in range(state_idx + 2, terminal_idx + 1):
                if not bool(market.exit_sellable[exit_idx, candidate]):
                    continue
                exit_price = float(market.exit_close_raw[exit_idx, candidate])
                value = math.log(
                    exit_price
                    * float(sell_multiplier[exit_idx - market.start_idx])
                    / (entry * buy_multiplier)
                ) + cash_value(exit_idx)
                result = max(result, value)
        return result

    entry = float(market.entry_open_raw[signal_idx + 1, symbol_idx])
    buy_value = -math.inf
    for exit_idx in range(signal_idx + 2, terminal_idx + 1):
        if not bool(market.exit_sellable[exit_idx, symbol_idx]):
            continue
        exit_price = float(market.exit_close_raw[exit_idx, symbol_idx])
        value = math.log(
            exit_price
            * float(sell_multiplier[exit_idx - market.start_idx])
            / (entry * buy_multiplier)
        ) + cash_value(exit_idx)
        buy_value = max(buy_value, value)
    return buy_value - cash_value(signal_idx + 1)


def test_study_distinguishes_coalescence_from_causal_availability() -> None:
    study = resolution.load_study()
    assert study["resolution"]["fixed_holding_horizon_used"] is False
    assert study["resolution"]["coalescence_is_causal_label_claim_allowed"] is False
    assert study["boundaries"]["training_performed"] is False


def test_cash_tree_first_common_state() -> None:
    parent = np.asarray([2, 3, 4, 4, 5, 5], dtype=np.int32)
    depth = np.asarray([3, 3, 2, 2, 1, 0], dtype=np.uint16)
    graph = resolution.CashPolicyGraph(
        holding_first_cash=np.empty((0, 0), dtype=np.int32),
        cash_parent=parent,
        cash_depth=depth,
        cash_up=(parent, parent[parent], parent[parent[parent[parent]]]),
        cash_log=np.zeros(6),
    )
    result = graph.first_common_cash(
        np.asarray([0, 0, 2, 5]), np.asarray([1, 2, 3, 5])
    )
    assert np.array_equal(result, np.asarray([4, 2, 4, 5]))


def test_resolution_decomposition_reconstructs_action_value(tmp_path) -> None:
    market = _market()
    solution = oracle.solve_relaxed_oracle(
        market=market,
        study=oracle.load_study(),
        cost_scenario="base",
        output_root=tmp_path,
        write_outputs=True,
    )
    record = _write_solution_arrays(tmp_path, solution)
    graph = resolution.build_cash_policy_graph(
        market=market, oracle_record=record
    )
    labels = pd.read_parquet(solution.label_files[0]["path"])
    resolved = resolution._resolve_label_frame(
        market=market,
        graph=graph,
        labels=labels,
        cost_scenario="base",
    )
    assert len(resolved) > 0
    assert bool((resolved["cash_resolution_delay"] >= 2).all())
    assert float(resolved["stored_value_identity_error"].abs().max()) < 1.0e-5


def test_truncated_prefix_solver_matches_exhaustive_intervals() -> None:
    market = _market()
    for signal_idx, symbol_idx, terminal_idx in ((0, 0, 3), (1, 1, 5), (2, 0, 5)):
        actual = resolution._prefix_terminal_action_advantage(
            market=market,
            cost_scenario="base",
            signal_date_idx=signal_idx,
            symbol_idx=symbol_idx,
            terminal_date_idx=terminal_idx,
        )
        expected = _brute_prefix_advantage(
            market,
            signal_idx=signal_idx,
            symbol_idx=symbol_idx,
            terminal_idx=terminal_idx,
            cost_scenario="base",
        )
        assert math.isclose(actual, expected, abs_tol=1.0e-12, rel_tol=0.0)
