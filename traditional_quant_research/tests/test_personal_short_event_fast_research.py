from __future__ import annotations

import pandas as pd
import pytest

from traditional_quant_research.experiments import personal_short_event_fast_research as fast


def test_prior_year_score_thresholds_use_only_earlier_prediction_years() -> None:
    frame = pd.DataFrame(
        {
            "eval_year": [2019, 2019, 2020, 2020, 2021, 2021],
            "score": [1.0, 2.0, 10.0, 20.0, 100.0, 200.0],
        }
    )

    thresholds = fast.prior_year_score_thresholds(frame, score_col="score", score_quantile=0.50)

    assert thresholds.iloc[0] == -float("inf")
    assert thresholds.iloc[2] == pytest.approx(1.5)
    assert thresholds.iloc[4] == pytest.approx(6.0)


def test_select_positions_can_refuse_days_and_honors_position_slots() -> None:
    frame = pd.DataFrame(
        [
            {
                "entry_date": pd.Timestamp("2024-01-02"),
                "exit_date": pd.Timestamp("2024-01-04"),
                "score": 2.0,
                "ml_score": 2.0,
                "target_net_ret_pct": 3.0,
                "code": "a",
            },
            {
                "entry_date": pd.Timestamp("2024-01-03"),
                "exit_date": pd.Timestamp("2024-01-05"),
                "score": 9.0,
                "ml_score": 9.0,
                "target_net_ret_pct": 5.0,
                "code": "b",
            },
            {
                "entry_date": pd.Timestamp("2024-01-05"),
                "exit_date": pd.Timestamp("2024-01-08"),
                "score": 1.0,
                "ml_score": 1.0,
                "target_net_ret_pct": 1.0,
                "code": "c",
            },
        ]
    )

    selected = fast.select_positions(frame, score_col="score", max_positions=1)

    assert selected["code"].tolist() == ["a", "c"]


def test_build_gate_mask_applies_risk_and_confirmation_filters() -> None:
    frame = pd.DataFrame(
        {
            "risk_flag_count": [1, 3],
            "risk_weak_market": [False, True],
            "risk_weak_industry": [False, False],
            "risk_high_open_gap": [False, False],
            "open_below_kama_break_limitup": [True, True],
            "close_cross_atr_upper": [True, False],
            "limit_up_run_ending_today": [1, 2],
        }
    )
    gate = fast.GateSpec(
        "test",
        max_risk_flags=2,
        block_weak_market=True,
        require_kama_atr=True,
        first_board_only=True,
    )

    mask = fast.build_gate_mask(frame, gate)

    assert mask.tolist() == [True, False]


def test_apply_score_floor_and_prior_quantile_filters_without_same_year_lookahead() -> None:
    frame = pd.DataFrame(
        {
            "eval_year": [2019, 2019, 2020, 2020],
            "score": [1.0, 2.0, 1.4, 1.6],
            "entry_date": pd.date_range("2019-01-01", periods=4),
            "exit_date": pd.date_range("2019-01-02", periods=4),
            "target_net_ret_pct": [1.0, 1.0, 1.0, 1.0],
            "ml_score": [1.0, 2.0, 1.4, 1.6],
        }
    )

    filtered = fast.apply_score_floor_and_prior_quantile(
        frame,
        score_col="score",
        score_floor=-float("inf"),
        score_quantile=0.50,
    )

    assert filtered["score"].tolist() == [1.0, 2.0, 1.6]
