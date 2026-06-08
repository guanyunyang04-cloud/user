from __future__ import annotations

import pandas as pd
import pytest

from traditional_quant_research.experiments import all_limitup_followup_research as follow


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "entry_date": pd.Timestamp("2024-01-03"),
                "exit_date": pd.Timestamp("2024-01-04"),
                "code": "a",
                "target_net_ret_pct": 5.0,
                "ml_score": 2.0,
                "predicted_ret_pct": 4.0,
                "predicted_big_loss_prob": 0.10,
                "entry_open_near_limit": False,
                "executable_entry": True,
                "open_below_kama_break_limitup": True,
                "close_cross_atr_upper": True,
                "one_word_limit_like": False,
                "near_one_word_limit_like": False,
                "limit_up_run_ending_today": 1,
                "board_stage": "first_board",
                "next_gap_bucket": "gap_0_to_3",
                "next_open_gap_pct": 2.0,
                "market_limitup_rate": 0.05,
                "market_breadth_5d": 0.60,
                "signal_amount_log10": 8.8,
                "volatility_20_pct": 3.0,
                "sell1_close_net_ret_pct": 5.0,
                "sell1_mfe_net_pct": 7.0,
                "sell1_mae_net_pct": -1.0,
                "sell3_close_net_ret_pct": 8.0,
                "sell3_mfe_net_pct": 10.0,
                "sell3_mae_net_pct": -2.0,
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "entry_date": pd.Timestamp("2024-01-04"),
                "exit_date": pd.Timestamp("2024-01-05"),
                "code": "b",
                "target_net_ret_pct": -6.0,
                "ml_score": 1.5,
                "predicted_ret_pct": 6.0,
                "predicted_big_loss_prob": 0.05,
                "entry_open_near_limit": False,
                "executable_entry": True,
                "open_below_kama_break_limitup": False,
                "close_cross_atr_upper": False,
                "one_word_limit_like": False,
                "near_one_word_limit_like": True,
                "limit_up_run_ending_today": 3,
                "board_stage": "third_board",
                "next_gap_bucket": "gap_ge_6",
                "next_open_gap_pct": 6.5,
                "market_limitup_rate": 0.01,
                "market_breadth_5d": 0.20,
                "signal_amount_log10": 7.2,
                "volatility_20_pct": 9.0,
                "signal_ret20_before_pct": 30.0,
                "price_position_60d": 0.95,
                "industry_ret5_mean": -2.0,
                "industry_limitup_rate": 0.0,
                "sell1_close_net_ret_pct": -6.0,
                "sell1_mfe_net_pct": 1.0,
                "sell1_mae_net_pct": -8.0,
                "sell3_close_net_ret_pct": -4.0,
                "sell3_mfe_net_pct": 2.0,
                "sell3_mae_net_pct": -9.0,
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "entry_date": pd.Timestamp("2024-01-04"),
                "exit_date": pd.Timestamp("2024-01-05"),
                "code": "c",
                "target_net_ret_pct": 2.0,
                "ml_score": 0.5,
                "predicted_ret_pct": 1.0,
                "predicted_big_loss_prob": 0.30,
                "entry_open_near_limit": True,
                "executable_entry": False,
                "open_below_kama_break_limitup": True,
                "close_cross_atr_upper": False,
                "one_word_limit_like": True,
                "near_one_word_limit_like": True,
                "limit_up_run_ending_today": 2,
                "board_stage": "second_board",
                "next_gap_bucket": "near_limit_up_open",
                "next_open_gap_pct": 9.2,
                "market_limitup_rate": 0.03,
                "market_breadth_5d": 0.40,
                "signal_amount_log10": 7.8,
                "volatility_20_pct": 5.0,
                "sell1_close_net_ret_pct": 2.0,
                "sell1_mfe_net_pct": 3.0,
                "sell1_mae_net_pct": -2.0,
                "sell3_close_net_ret_pct": 3.0,
                "sell3_mfe_net_pct": 5.0,
                "sell3_mae_net_pct": -1.0,
            },
        ]
    )


def test_assign_event_facets_splits_limitup_types() -> None:
    facets = follow.assign_event_facets(_frame())

    assert facets["board_run_bucket"].tolist() == ["run_1", "run_3", "run_2"]
    assert facets["one_word_type"].tolist() == ["regular_board", "near_one_word", "one_word"]
    assert facets["kama_atr_type"].tolist() == ["kama_atr", "neither", "kama_only"]


def test_stress_adjustments_filter_and_haircut() -> None:
    stressed, meta = follow.apply_stress_adjustments(
        _frame(),
        {
            "scenario": "test",
            "block_non_executable": True,
            "block_near_limit": True,
            "extra_fee_bps": 70.0,
            "gap_haircuts": ((5.0, 50.0),),
            "tail_quantile": 0.50,
            "tail_loss_multiplier": 2.0,
        },
    )

    assert meta["candidate_count_before_stress"] == 3
    assert meta["blocked_by_stress_count"] == 1
    assert stressed["code"].tolist() == ["a", "b"]
    assert stressed.loc[stressed["code"].eq("a"), "target_net_ret_pct"].iloc[0] == pytest.approx(4.3)
    assert stressed.loc[stressed["code"].eq("b"), "target_net_ret_pct"].iloc[0] == pytest.approx(-14.4)


def test_label_path_summary_keeps_selected_and_pool_groups() -> None:
    frame = _frame()
    frame["selected_executable_pos1"] = [True, True, False]
    frame["selected_executable_pos2"] = [True, True, False]

    summary = follow.summarize_label_paths(frame, label_windows=(1, 3))
    selected_1d = summary.loc[
        summary["group_name"].eq("selected_executable_pos1") & summary["horizon"].eq(1)
    ].iloc[0]

    assert selected_1d["event_count"] == 2
    assert selected_1d["mean_close_net_ret_pct"] == pytest.approx(-0.5)
    assert selected_1d["big_loss_rate"] == pytest.approx(0.5)


def test_failure_attribution_adds_explainable_reasons() -> None:
    frame = _frame()
    failures = frame.loc[frame["code"].eq("b")].copy()

    attributed = follow.attribute_failure_reasons(failures, reference=frame)

    row = attributed.iloc[0]
    assert bool(row["reason_weak_market"]) is True
    assert bool(row["reason_high_open_gap"]) is True
    assert bool(row["reason_advanced_board"]) is True
    assert bool(row["reason_model_overconfidence"]) is True


def test_feature_family_specs_classifies_open_and_market_features() -> None:
    specs = follow.feature_family_specs(
        (
            "market_breadth_5d",
            "industry_ret5_mean",
            "next_open_gap_pct",
            "entry_open_near_limit",
            "signal_amount_log10",
            "turn",
            "kama_slope_pct",
        )
    )

    assert specs["market_industry"] == ("market_breadth_5d", "industry_ret5_mean")
    assert specs["open_print"] == ("next_open_gap_pct", "entry_open_near_limit")
    assert specs["signal_liquidity"] == ("signal_amount_log10", "turn")
    assert specs["trend_risk"] == ("kama_slope_pct",)
