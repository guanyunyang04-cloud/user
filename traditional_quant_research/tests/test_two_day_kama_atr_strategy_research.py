from __future__ import annotations

import pandas as pd
import pytest

from traditional_quant_research.experiments.two_day_kama_atr_strategy_research import (
    StrategySpec,
    build_day1_candidate_trades_from_feature_panel,
    entry_filter_mask,
    exit_for_policy,
    materialize_strategy_trades,
    schedule_max_positions,
)


def _strategy_feature_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-02", periods=6)
    rows = []
    specs = {
        "000001.SZ": {
            "closes": [10.0, 11.0, 12.1, 13.0, 13.8, 13.5],
            "day2_limit": True,
            "day2_atr_upper": 12.0,
            "day1_amount": 200_000_000.0,
        },
        "000002.SZ": {
            "closes": [20.0, 22.0, 21.5, 20.8, 20.0, 19.5],
            "day2_limit": False,
            "day2_atr_upper": 25.0,
            "day1_amount": 100_000_000.0,
        },
    }
    for code, spec in specs.items():
        for idx, (date, close) in enumerate(zip(dates, spec["closes"])):
            is_day1 = idx == 1
            is_day2 = idx == 2
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "date_index": idx,
                    "name_on_date": code,
                    "industry": "test",
                    "open": close if not is_day2 else close * 0.98,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "close": close,
                    "pctChg": 10.0 if is_day1 or (is_day2 and spec["day2_limit"]) else -2.0,
                    "amount": spec["day1_amount"] if is_day1 else 80_000_000.0,
                    "turn": 3.0,
                    "kama": 10.5 if code == "000001.SZ" else 21.0,
                    "atr_upper": 12.5 if is_day1 else (spec["day2_atr_upper"] if is_day2 else close * 1.2),
                    "limit_up_like": bool(is_day1 or (is_day2 and spec["day2_limit"])),
                    "one_word_limit_like": False,
                    "near_one_word_limit_like": False,
                    "limit_up_run_ending_today": 1 if is_day1 else (2 if is_day2 and spec["day2_limit"] else 0),
                    "board_stage": "first_board" if is_day1 else ("second_board" if is_day2 and spec["day2_limit"] else "none"),
                    "signal_close_position": 0.8 if is_day1 else 0.5,
                }
            )
    return pd.DataFrame(rows)


def test_day1_candidates_keep_confirmed_and_nonconfirmed_paths() -> None:
    candidates, path = build_day1_candidate_trades_from_feature_panel(
        _strategy_feature_panel(),
        signal_start_date="2026-01-01",
        signal_end_date="2026-01-31",
        max_hold_offset=3,
    )

    assert candidates["code"].tolist() == ["000001.SZ", "000002.SZ"]
    confirmed = candidates.set_index("code")["confirmed_two_day_breakout"].to_dict()
    assert bool(confirmed["000001.SZ"]) is True
    assert bool(confirmed["000002.SZ"]) is False
    assert set(path["future_offset"]) == {1, 2, 3}
    first = candidates.loc[candidates["code"].eq("000001.SZ")].iloc[0]
    assert first["off1_close_ret_pct"] == pytest.approx(((13.0 / (12.1 * 0.98)) - 1.0) * 100.0)


def test_exit_policy_cuts_nonconfirmed_at_day3_open() -> None:
    candidates, _ = build_day1_candidate_trades_from_feature_panel(
        _strategy_feature_panel(),
        signal_start_date="2026-01-01",
        signal_end_date="2026-01-31",
        max_hold_offset=3,
    )
    nonconfirmed = candidates.loc[candidates["code"].eq("000002.SZ")].iloc[0]
    exit_date, gross_ret, reason = exit_for_policy(nonconfirmed, "confirm_day3_close_nonconfirm_day3_open")

    assert exit_date == pd.Timestamp("2026-01-07")
    assert reason == "nonconfirm_day3_open"
    assert gross_ret == pytest.approx(nonconfirmed["off1_open_ret_pct"])


def test_materialize_strategy_applies_fees_and_max_positions() -> None:
    candidates, _ = build_day1_candidate_trades_from_feature_panel(
        _strategy_feature_panel(),
        signal_start_date="2026-01-01",
        signal_end_date="2026-01-31",
        max_hold_offset=3,
    )
    spec = StrategySpec(
        entry_filter="fillable_open",
        exit_policy="confirm_day3_close_nonconfirm_day3_open",
        score_column="day1_amount_log10",
        max_positions=1,
    )

    trades = materialize_strategy_trades(candidates, spec=spec, fee_bps=30.0)

    assert len(trades) == 1
    assert trades.iloc[0]["code"] == "000001.SZ"
    assert trades.iloc[0]["net_ret_pct"] == pytest.approx(trades.iloc[0]["gross_ret_pct"] - 0.30)


def test_refined_entry_filters_use_only_day2_open_known_fields() -> None:
    frame = pd.DataFrame(
        {
            "day2_open_gap_pct": [-4.0, -1.0, 1.0, -6.0],
            "day1_close_vs_atr_upper_pct": [-11.0, -6.0, -12.0, -12.0],
            "day1_board_stage": ["first_board", "second_board", "second_board", "second_board"],
        }
    )

    assert entry_filter_mask(frame, "gap_m5_m3").tolist() == [True, False, False, False]
    assert entry_filter_mask(frame, "gap_le0_atr_below_m10").tolist() == [True, False, False, True]
    assert entry_filter_mask(frame, "gap_le0_second_atr_below_m5").tolist() == [False, True, False, True]


def test_schedule_max_positions_blocks_overlapping_close_exits() -> None:
    frame = pd.DataFrame(
        [
            {
                "code": "a",
                "entry_date": pd.Timestamp("2026-01-05"),
                "exit_date": pd.Timestamp("2026-01-06"),
                "score": 10.0,
            },
            {
                "code": "b",
                "entry_date": pd.Timestamp("2026-01-06"),
                "exit_date": pd.Timestamp("2026-01-07"),
                "score": 9.0,
            },
            {
                "code": "c",
                "entry_date": pd.Timestamp("2026-01-07"),
                "exit_date": pd.Timestamp("2026-01-08"),
                "score": 8.0,
            },
        ]
    )

    selected = schedule_max_positions(frame, score_column="score", max_positions=1)

    assert selected["code"].tolist() == ["a", "c"]
