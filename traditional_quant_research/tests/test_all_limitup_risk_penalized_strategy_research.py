from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from traditional_quant_research.experiments import all_limitup_risk_penalized_strategy_research as risk


def _reference() -> pd.DataFrame:
    rows = []
    for year in (2022, 2023, 2024):
        rows.extend(
            [
                {
                    "year": year,
                    "market_breadth_5d": 0.20,
                    "market_limitup_rate": 0.01,
                    "industry_ret5_mean": -2.0,
                    "industry_limitup_rate": 0.00,
                    "signal_ret20_before_pct": 5.0,
                    "price_position_60d": 0.20,
                    "volatility_20_pct": 3.0,
                    "signal_amount_log10": 7.0,
                },
                {
                    "year": year,
                    "market_breadth_5d": 0.50,
                    "market_limitup_rate": 0.04,
                    "industry_ret5_mean": 0.0,
                    "industry_limitup_rate": 0.03,
                    "signal_ret20_before_pct": 20.0,
                    "price_position_60d": 0.80,
                    "volatility_20_pct": 8.0,
                    "signal_amount_log10": 9.0,
                },
            ]
        )
    return pd.DataFrame(rows)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "eval_year": 2024,
                "year": 2024,
                "entry_date": pd.Timestamp("2024-01-03"),
                "exit_date": pd.Timestamp("2024-01-04"),
                "code": "a",
                "name_on_date": "A",
                "ml_score": 2.0,
                "predicted_ret_pct": 3.0,
                "predicted_big_loss_prob": 0.10,
                "target_net_ret_pct": 5.0,
                "entry_open_near_limit": False,
                "executable_entry": True,
                "open_below_kama_break_limitup": True,
                "close_cross_atr_upper": True,
                "next_open_gap_pct": 2.0,
                "market_breadth_5d": 0.60,
                "market_limitup_rate": 0.05,
                "industry_ret5_mean": 1.0,
                "industry_limitup_rate": 0.05,
                "signal_ret20_before_pct": 5.0,
                "price_position_60d": 0.30,
                "volatility_20_pct": 3.0,
                "limit_up_run_ending_today": 1,
                "signal_amount_log10": 8.5,
                "liquid_amount_ok": True,
            },
            {
                "date": pd.Timestamp("2024-01-02"),
                "eval_year": 2024,
                "year": 2024,
                "entry_date": pd.Timestamp("2024-01-03"),
                "exit_date": pd.Timestamp("2024-01-04"),
                "code": "b",
                "name_on_date": "B",
                "ml_score": 1.8,
                "predicted_ret_pct": 4.0,
                "predicted_big_loss_prob": 0.40,
                "target_net_ret_pct": -6.0,
                "entry_open_near_limit": False,
                "executable_entry": True,
                "open_below_kama_break_limitup": False,
                "close_cross_atr_upper": False,
                "next_open_gap_pct": 6.0,
                "market_breadth_5d": 0.10,
                "market_limitup_rate": 0.005,
                "industry_ret5_mean": -3.0,
                "industry_limitup_rate": 0.0,
                "signal_ret20_before_pct": 30.0,
                "price_position_60d": 0.95,
                "volatility_20_pct": 10.0,
                "limit_up_run_ending_today": 4,
                "signal_amount_log10": 6.8,
                "liquid_amount_ok": False,
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "eval_year": 2024,
                "year": 2024,
                "entry_date": pd.Timestamp("2024-01-05"),
                "exit_date": pd.Timestamp("2024-01-08"),
                "code": "c",
                "name_on_date": "C",
                "ml_score": 0.2,
                "predicted_ret_pct": 1.0,
                "predicted_big_loss_prob": 0.05,
                "target_net_ret_pct": 2.0,
                "entry_open_near_limit": False,
                "executable_entry": True,
                "open_below_kama_break_limitup": True,
                "close_cross_atr_upper": False,
                "next_open_gap_pct": -1.0,
                "market_breadth_5d": 0.55,
                "market_limitup_rate": 0.03,
                "industry_ret5_mean": 0.5,
                "industry_limitup_rate": 0.02,
                "signal_ret20_before_pct": 10.0,
                "price_position_60d": 0.40,
                "volatility_20_pct": 4.0,
                "limit_up_run_ending_today": 1,
                "signal_amount_log10": 8.0,
                "liquid_amount_ok": True,
            },
        ]
    )


def test_build_risk_flags_uses_prior_year_thresholds() -> None:
    flagged = risk.build_risk_flags(_frame(), _reference(), eval_years=(2024,))
    weak = flagged.loc[flagged["code"].eq("b")].iloc[0]
    clean = flagged.loc[flagged["code"].eq("a")].iloc[0]

    assert bool(weak["risk_weak_market"]) is True
    assert bool(weak["risk_weak_industry"]) is True
    assert bool(weak["risk_overextended"]) is True
    assert bool(weak["risk_high_volatility"]) is True
    assert bool(weak["risk_no_kama_atr_confirmation"]) is True
    assert weak["risk_flag_count"] >= 7
    assert bool(clean["risk_weak_market"]) is False
    assert bool(clean["risk_no_kama_atr_confirmation"]) is False


def test_penalty_score_subtracts_flags_and_big_loss_probability() -> None:
    flagged = risk.build_risk_flags(_frame(), _reference(), eval_years=(2024,))
    scored = risk.build_penalty_scores(
        flagged,
        (
            {
                "penalty_name": "test",
                "weights": {"risk_weak_market": 1.0, "risk_high_open_gap": 0.5},
                "big_loss_prob_weight": 2.0,
            },
        ),
    )

    a = scored.loc[scored["code"].eq("a"), "risk_score_test"].iloc[0]
    b = scored.loc[scored["code"].eq("b"), "risk_score_test"].iloc[0]

    assert a == pytest.approx(1.8)
    assert b == pytest.approx(-0.5)


def test_select_risk_adjusted_positions_can_skip_original_top1_for_fallback() -> None:
    flagged = risk.build_risk_flags(_frame(), _reference(), eval_years=(2024,))
    scored = risk.build_penalty_scores(
        flagged,
        (
            {
                "penalty_name": "strict",
                "weights": {"risk_weak_market": 2.0, "risk_high_open_gap": 1.0},
                "big_loss_prob_weight": 2.0,
            },
        ),
    )

    selected = risk.select_risk_adjusted_positions(
        scored,
        score_col="risk_score_strict",
        min_score=0.0,
        max_original_rank=3,
        max_positions=1,
    )

    assert selected["code"].tolist() == ["a", "c"]


def test_strategy_id_formats_thresholds_and_rank_caps() -> None:
    assert risk.strategy_id_for(
        penalty_name="balanced",
        min_score=-np.inf,
        max_original_rank=None,
        max_positions=1,
    ) == "balanced__all__score_none__pos1"
    assert risk.strategy_id_for(
        penalty_name="balanced",
        min_score=-0.5,
        max_original_rank=3,
        max_positions=2,
    ) == "balanced__scan3__score_m0p5__pos2"


def test_stress_selected_applies_extra_fee_and_tail_multiplier() -> None:
    stressed = risk.stress_selected(
        _frame(),
        {
            "scenario": "test",
            "extra_fee_bps": 70.0,
            "tail_quantile": 0.50,
            "tail_loss_multiplier": 2.0,
        },
    )

    assert stressed.loc[stressed["code"].eq("a"), "target_net_ret_pct"].iloc[0] == pytest.approx(4.3)
    assert stressed.loc[stressed["code"].eq("b"), "target_net_ret_pct"].iloc[0] == pytest.approx(-13.4)
