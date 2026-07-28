from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy.seq100_candidate_execution import (
    evaluate_candidate_execution,
)


def _manifest(*, forward_days: int = 3, tail_days: int = 1, recovery: float = 0.0) -> dict:
    return {
        "forward_days": forward_days,
        "execution_tail_days": tail_days,
        "execution_costs": {
            "lot_size": 100,
            "commission_bps": 3.0,
            "minimum_commission_cny": 5.0,
            "stamp_tax_bps": 5.0,
            "transfer_fee_bps": 0.1,
            "slippage_bps": 7.0,
            "stress_slippage_multiplier": 2.0,
            "stamp_tax_schedule": [
                {"effective_date": "1900-01-01", "stamp_tax_bps": 10.0},
                {"effective_date": "2023-08-28", "stamp_tax_bps": 5.0},
            ],
            "commission_sides": ["buy", "sell"],
            "minimum_commission_applied_per_order": True,
            "stamp_tax_sides": ["sell"],
            "transfer_fee_sides": ["buy", "sell"],
            "slippage_application": "buy_price*(1+bps/10000), sell_price*(1-bps/10000)",
            "unaffordable_or_unfilled_order": "retain_cash",
        },
        "terminal_execution": {
            "unresolved_after_tail": "apply_precommitted_recovery_fraction",
            "recovery_fraction_of_entry_notional": recovery,
        },
    }


def _candidate_frame(**overrides) -> pd.DataFrame:
    values = {
        "trade_date": ["2024-01-02"],
        "symbol": ["000001.SZ"],
        "score": [1.0],
        "predicted_exit_day": [2.0],
        "entry_filled": [True],
        "entry_open_raw": [10.0],
        "realized_plan_value": [0.15],
    }
    values.update(overrides)
    return pd.DataFrame(values)


def _date_path(rows: int, dates: tuple[str, ...] = ("2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08")) -> np.ndarray:
    return np.asarray([dates for _ in range(rows)], dtype=object)


def test_unfilled_entry_retains_cash_and_candidate_row() -> None:
    frame = _candidate_frame(entry_filled=[False])
    result = evaluate_candidate_execution(
        frame,
        np.asarray([[11.0, 12.0, 13.0, 14.0]]),
        np.ones((1, 4), dtype=bool),
        manifest=_manifest(),
        exit_trade_date_path=_date_path(1),
        top_k_values=(1,),
    )

    assert len(result.candidates) == 1
    row = result.candidates.iloc[0]
    assert row["realized_plan_exit_status"] == "entry_unfilled_cash"
    assert row["gross_realized_plan_return"] == 0.0
    assert row["net_realized_plan_return_base"] == 0.0
    assert row["shares"] == 0
    assert row["ending_cash_base_cny"] == 1_000_000.0
    assert bool(row["realized_plan_covered"])


def test_t_plus_one_clamps_day_one_and_supports_global_date_values() -> None:
    frame = _candidate_frame(predicted_exit_day=[1.0], signal_date_idx=[0])
    result = evaluate_candidate_execution(
        frame,
        np.asarray([[11.0, 12.0, 13.0, 14.0]]),
        np.ones((1, 4), dtype=bool),
        manifest=_manifest(),
        date_values=("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"),
        top_k_values=(1,),
    )

    row = result.candidates.iloc[0]
    assert row["realized_plan_planned_exit_day"] == 2
    assert row["realized_plan_resolved_exit_day"] == 2
    assert row["realized_plan_resolved_exit_date"] == "2024-01-04"
    assert row["gross_realized_plan_return"] == pytest.approx(0.2)
    assert row["net_realized_plan_value_base"] == pytest.approx(
        row["realized_plan_value"]
        + row["net_realized_plan_return_base"]
        - row["gross_realized_plan_return"]
    )


def test_blocked_close_defers_through_execution_tail() -> None:
    result = evaluate_candidate_execution(
        _candidate_frame(),
        np.asarray([[11.0, 12.0, 13.0, 14.0]]),
        np.asarray([[True, False, False, True]]),
        manifest=_manifest(),
        exit_trade_date_path=_date_path(1),
        top_k_values=(1,),
    )

    row = result.candidates.iloc[0]
    assert row["realized_plan_planned_exit_day"] == 2
    assert row["realized_plan_resolved_exit_day"] == 4
    assert row["realized_plan_exit_status"] == "filled_deferred_exit"
    assert row["gross_realized_plan_return"] == pytest.approx(0.4)


def test_unresolved_exit_uses_zero_terminal_recovery() -> None:
    result = evaluate_candidate_execution(
        _candidate_frame(),
        np.asarray([[11.0, 12.0, 13.0, 14.0]]),
        np.zeros((1, 4), dtype=bool),
        manifest=_manifest(recovery=0.0),
        exit_trade_date_path=_date_path(1),
        top_k_values=(1,),
    )

    row = result.candidates.iloc[0]
    assert row["realized_plan_exit_status"] == "terminal_recovery"
    assert bool(row["realized_plan_terminal_recovery"])
    assert row["realized_plan_resolved_exit_day"] == 4
    assert row["gross_realized_plan_return"] == -1.0
    assert bool(row["realized_plan_covered"])


def test_stamp_tax_uses_resolved_exit_trade_date() -> None:
    frame = _candidate_frame(
        trade_date=["2023-08-23", "2023-08-24"],
        symbol=["000001.SZ", "000002.SZ"],
        score=[2.0, 1.0],
        predicted_exit_day=[2.0, 2.0],
        entry_filled=[True, True],
        entry_open_raw=[10.0, 10.0],
        realized_plan_value=[0.0, 0.0],
    )
    dates = np.asarray(
        [
            ["2023-08-24", "2023-08-25"],
            ["2023-08-25", "2023-08-28"],
        ],
        dtype=object,
    )
    result = evaluate_candidate_execution(
        frame,
        np.full((2, 2), 10.0),
        np.ones((2, 2), dtype=bool),
        manifest=_manifest(forward_days=2, tail_days=0),
        exit_trade_date_path=dates,
        top_k_values=(1,),
    )

    before, after = result.candidates.iloc[0], result.candidates.iloc[1]
    assert after["execution_cost_base_cny"] < before["execution_cost_base_cny"]
    assert after["net_realized_plan_return_base"] > before["net_realized_plan_return_base"]


def test_lot_rounding_minimum_fee_and_unaffordable_cash() -> None:
    frame = _candidate_frame(
        trade_date=["2024-01-02", "2024-01-02"],
        symbol=["000001.SZ", "000002.SZ"],
        score=[2.0, 1.0],
        predicted_exit_day=[2.0, 2.0],
        entry_filled=[True, True],
        entry_open_raw=[100.0, 101.0],
        realized_plan_value=[0.0, 0.0],
    )
    result = evaluate_candidate_execution(
        frame,
        np.asarray([[100.0, 101.0], [101.0, 102.0]]),
        np.ones((2, 2), dtype=bool),
        manifest=_manifest(forward_days=2, tail_days=0),
        exit_trade_date_path=_date_path(2, ("2024-01-03", "2024-01-04")),
        top_k_values=(1,),
        daily_cohort_cash_cny=10_050.0,
    )

    affordable, unaffordable = result.candidates.iloc[0], result.candidates.iloc[1]
    assert affordable["shares"] == 100
    assert affordable["shares"] % 100 == 0
    assert affordable["execution_cost_base_cny"] > 10.0
    assert unaffordable["shares"] == 0
    assert unaffordable["net_realized_plan_return_base"] == 0.0
    assert unaffordable["ending_cash_base_cny"] == 10_050.0


def test_topk_retains_every_selected_name_and_cash_slot() -> None:
    frame = _candidate_frame(
        trade_date=["2024-01-02", "2024-01-02"],
        symbol=["000001.SZ", "000002.SZ"],
        score=[1.0, np.nan],
        predicted_exit_day=[2.0, 2.0],
        entry_filled=[False, True],
        entry_open_raw=[10.0, 10.0],
        realized_plan_value=[np.nan, -1.0],
    )
    manifest = _manifest(forward_days=2, tail_days=0, recovery=0.0)
    result = evaluate_candidate_execution(
        frame,
        np.full((2, 2), 10.0),
        np.zeros((2, 2), dtype=bool),
        manifest=manifest,
        exit_trade_date_path=_date_path(2, ("2024-01-03", "2024-01-04")),
        top_k_values=(1, 3),
    )

    assert len(result.candidates) == 2
    top1 = result.daily_topk.loc[result.daily_topk["top_k"].eq(1)].iloc[0]
    assert top1["selected_symbols"] == ["000001.SZ"]
    assert top1["selected_entry_fill_rate"] == 0.0
    assert top1["selected_net_realized_plan_return_base"] == 0.0
    assert top1["selected_realized_plan_value_coverage"] == 0.0
    assert pd.isna(top1["selected_net_realized_plan_value_base"])
    assert pd.isna(result.candidates.iloc[0]["net_realized_plan_value_base"])
    topk = result.daily_topk.loc[result.daily_topk["top_k"].eq(3)].iloc[0]
    assert topk["selected_symbols"] == ["000001.SZ", "000002.SZ"]
    assert topk["selected_count"] == 2
    assert topk["unused_cash_slot_count"] == 1
    assert topk["selected_realized_plan_coverage"] == 1.0
    assert topk["selected_cash_retained_count_base"] == 2
    assert topk["selected_terminal_recovery_count_base"] == 1
    assert result.execution_costs.lot_size == 100
    assert result.execution_costs.slippage_bps == 7.0
    assert topk["selected_net_realized_plan_return_stress"] <= topk["selected_net_realized_plan_return_base"]
