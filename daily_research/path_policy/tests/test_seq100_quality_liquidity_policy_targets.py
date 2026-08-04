from __future__ import annotations

import math

import numpy as np
import pandas as pd

from daily_research.path_policy import (
    seq100_quality_liquidity_policy_targets as policy,
)


def _panels(*, days: int = 32, symbols: int = 4):
    daily = np.full((days, symbols, 4), 10.0, dtype=np.float32)
    daily[:, :, 1] = 10.2
    daily[:, :, 2] = 9.8
    entry = np.ones((days, symbols), dtype=bool)
    sellable = np.ones((days, symbols), dtype=bool)
    return daily, entry, sellable


def _next_sellable(sellable: np.ndarray) -> np.ndarray:
    days, symbols = sellable.shape
    output = np.full((days + 1, symbols), -1, dtype=np.int32)
    next_idx = np.full(symbols, -1, dtype=np.int32)
    for day in range(days - 1, -1, -1):
        next_idx[sellable[day]] = day
        output[day] = next_idx
    return output


def test_config_and_task_inventory_include_d10_and_d20() -> None:
    config = policy._load_config()
    tasks = policy._target_plan()

    assert tuple(config["target_contract"]["timeout_days"]) == (10, 20)
    assert len(tasks) == 18
    assert {task["horizon"] for task in tasks} == {10, 20}
    assert len(policy._replay_specs()) == 32
    assert all(task["variant"] == "compact_core" for task in tasks)


def test_policy_outcome_uses_d2_intraday_and_blocked_fill_semantics() -> None:
    daily, entry, sellable = _panels()
    signal_idx = 1
    daily[signal_idx + 1, :, 1] = 20.0  # D1 is intentionally ignored.
    daily[signal_idx + 2, 0, 1] = 10.8
    daily[signal_idx + 2, 1, 0] = 10.9
    daily[signal_idx + 2, 1, 1] = 11.0
    daily[signal_idx + 2, 2, 1] = 10.9
    sellable[signal_idx + 2, 2] = False
    daily[signal_idx + 3, 2, 0] = 9.5
    daily[signal_idx + 10, 3, 3] = 8.5

    outcome = policy._resolve_batch_outcome(
        signal_idx=signal_idx,
        symbols=np.arange(4, dtype=np.int32),
        horizon=10,
        cutoff_idx=31,
        daily_raw=daily,
        entry_filled=entry,
        exit_sellable=sellable,
        next_open_sellable_idx=_next_sellable(sellable),
    )

    assert outcome.outcome_code.tolist() == [1, 1, 1, 2]
    assert outcome.fill_phase.tolist() == [2, 1, 3, 4]
    assert outcome.trigger_day.tolist() == [2, 2, 2, 10]
    assert outcome.fill_day.tolist() == [2, 2, 3, 10]
    assert np.all(outcome.tp_valid)
    assert outcome.tp_hit.tolist() == [1.0, 1.0, 1.0, 0.0]
    assert math.isclose(float(outcome.policy_return[0]), 0.08, abs_tol=1e-6)
    assert math.isclose(float(outcome.policy_return[1]), 0.09, abs_tol=1e-6)
    assert math.isclose(float(outcome.policy_return[2]), -0.05, abs_tol=1e-6)
    assert math.isclose(float(outcome.timeout_return[3]), -0.15, abs_tol=1e-6)


def test_d10_timeout_and_d20_take_profit_are_distinct_targets() -> None:
    daily, entry, sellable = _panels(symbols=1)
    signal_idx = 1
    daily[signal_idx + 10, 0, 3] = 9.7
    daily[signal_idx + 15, 0, 1] = 10.8
    next_idx = _next_sellable(sellable)

    d10 = policy._resolve_batch_outcome(
        signal_idx=signal_idx,
        symbols=np.array([0], dtype=np.int32),
        horizon=10,
        cutoff_idx=31,
        daily_raw=daily,
        entry_filled=entry,
        exit_sellable=sellable,
        next_open_sellable_idx=next_idx,
    )
    d20 = policy._resolve_batch_outcome(
        signal_idx=signal_idx,
        symbols=np.array([0], dtype=np.int32),
        horizon=20,
        cutoff_idx=31,
        daily_raw=daily,
        entry_filled=entry,
        exit_sellable=sellable,
        next_open_sellable_idx=next_idx,
    )

    assert d10.outcome_code[0] == policy.OUTCOME_TIMEOUT
    assert d20.outcome_code[0] == policy.OUTCOME_TAKE_PROFIT
    assert d10.tp_hit[0] == 0.0
    assert d20.tp_hit[0] == 1.0


def test_blocked_timeout_fills_at_next_sellable_open() -> None:
    daily, entry, sellable = _panels(symbols=1)
    signal_idx = 1
    sellable[signal_idx + 10, 0] = False
    daily[signal_idx + 12, 0, 0] = 8.0

    outcome = policy._resolve_batch_outcome(
        signal_idx=signal_idx,
        symbols=np.array([0], dtype=np.int32),
        horizon=10,
        cutoff_idx=31,
        daily_raw=daily,
        entry_filled=entry,
        exit_sellable=sellable,
        next_open_sellable_idx=_next_sellable(sellable),
    )

    assert outcome.outcome_code[0] == policy.OUTCOME_TIMEOUT
    assert outcome.trigger_day[0] == 10
    assert outcome.fill_day[0] == 11
    assert outcome.fill_phase[0] == policy.FILL_DELAYED_OPEN
    assert math.isclose(float(outcome.timeout_return[0]), 0.0, abs_tol=1e-7)


def test_incomplete_horizon_and_unfilled_entry_are_invalid() -> None:
    daily, entry, sellable = _panels(symbols=2)
    entry[1, 1] = False

    incomplete = policy._resolve_batch_outcome(
        signal_idx=15,
        symbols=np.array([0], dtype=np.int32),
        horizon=20,
        cutoff_idx=31,
        daily_raw=daily,
        entry_filled=entry,
        exit_sellable=sellable,
        next_open_sellable_idx=_next_sellable(sellable),
    )
    unfilled = policy._resolve_batch_outcome(
        signal_idx=1,
        symbols=np.array([1], dtype=np.int32),
        horizon=10,
        cutoff_idx=31,
        daily_raw=daily,
        entry_filled=entry,
        exit_sellable=sellable,
        next_open_sellable_idx=_next_sellable(sellable),
    )

    assert not incomplete.tp_valid[0]
    assert not incomplete.endpoint_valid[0]
    assert not unfilled.tp_valid[0]
    assert not unfilled.endpoint_valid[0]


def test_binary_metrics_preserve_raw_score_ranking() -> None:
    actual = np.array([0, 0, 1, 1], dtype=np.float32)
    score = np.array([-2.0, -1.0, 0.1, 2.0], dtype=np.float32)

    _, metrics = policy._binary_daily_metrics(
        date_idx=np.ones(4, dtype=np.int32),
        actual=actual,
        prediction=score,
        fractions=(0.5,),
    )

    assert metrics["roc_auc"] == 1.0
    assert metrics["top50_hit_rate"] == 1.0
    assert metrics["top50_capture"] == 1.0


def test_unit_time_metrics_use_only_completed_sells() -> None:
    trades = pd.DataFrame(
        {
            "side": ["sell", "sell", "buy"],
            "status": ["filled", "filled", "filled"],
            "net_return": [0.10, -0.05, 0.0],
            "holding_days": [5, 10, 0],
        }
    )

    metrics = policy._unit_time_metrics(trades, slot_count=6)

    assert metrics["completed_trade_count"] == 2
    assert math.isclose(
        metrics["mean_net_return_per_holding_day"], 0.0075, abs_tol=1e-12
    )
    assert math.isclose(metrics["completed_trades_per_slot_year"], 1 / 9)
