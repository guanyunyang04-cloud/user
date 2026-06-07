from __future__ import annotations

import pandas as pd

from traditional_quant_research.experiments.all_limitup_ml_strategy_research import (
    build_pool_masks,
    build_walk_forward_plan,
    schedule_top_positions,
    summarize_portfolio,
)


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "entry_date": pd.Timestamp("2024-01-03"),
                "exit_date": pd.Timestamp("2024-01-04"),
                "code": "a",
                "ml_score": 0.8,
                "target_net_ret_pct": 5.0,
                "executable_entry": True,
                "entry_open_near_limit": False,
                "open_below_kama_break_limitup": True,
                "close_cross_atr_upper": False,
            },
            {
                "date": pd.Timestamp("2024-01-02"),
                "entry_date": pd.Timestamp("2024-01-03"),
                "exit_date": pd.Timestamp("2024-01-04"),
                "code": "b",
                "ml_score": 0.7,
                "target_net_ret_pct": -2.0,
                "executable_entry": True,
                "entry_open_near_limit": False,
                "open_below_kama_break_limitup": False,
                "close_cross_atr_upper": False,
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "entry_date": pd.Timestamp("2024-01-04"),
                "exit_date": pd.Timestamp("2024-01-05"),
                "code": "c",
                "ml_score": 0.9,
                "target_net_ret_pct": 7.0,
                "executable_entry": True,
                "entry_open_near_limit": False,
                "open_below_kama_break_limitup": True,
                "close_cross_atr_upper": True,
            },
            {
                "date": pd.Timestamp("2024-01-04"),
                "entry_date": pd.Timestamp("2024-01-05"),
                "exit_date": pd.Timestamp("2024-01-08"),
                "code": "d",
                "ml_score": 0.1,
                "target_net_ret_pct": 1.0,
                "executable_entry": False,
                "entry_open_near_limit": True,
                "open_below_kama_break_limitup": True,
                "close_cross_atr_upper": True,
            },
        ]
    )


def test_walk_forward_plan_uses_only_prior_years() -> None:
    events = pd.DataFrame(
        {
            "year": [2020, 2021, 2022, 2023],
            "date": pd.to_datetime(["2020-01-02", "2021-01-02", "2022-01-02", "2023-01-02"]),
            "target_net_ret_pct": [1.0, 2.0, 3.0, 4.0],
        }
    )

    plan = build_walk_forward_plan(events, eval_years=(2022, 2023), max_train_years=2, min_train_years=1)

    assert plan.loc[plan["eval_year"].eq(2022), "train_years"].iloc[0] == "2020,2021"
    assert plan.loc[plan["eval_year"].eq(2023), "train_years"].iloc[0] == "2021,2022"
    assert not plan["train_years"].str.contains("2023").iloc[0]


def test_schedule_top_positions_blocks_same_day_exit_overlap() -> None:
    selected = schedule_top_positions(_prediction_frame(), pool_name="all_open_known", max_positions=1)

    assert selected["code"].tolist() == ["a", "d"]


def test_pool_masks_compare_full_pool_and_kama_pool() -> None:
    frame = _prediction_frame()
    masks = build_pool_masks(frame)

    assert masks["all_open_known"].tolist() == [True, True, True, True]
    assert masks["executable_only"].tolist() == [True, True, True, False]
    assert masks["kama_break_hard"].tolist() == [True, False, True, True]
    assert masks["kama_atr_hard"].tolist() == [False, False, True, True]


def test_summarize_portfolio_uses_period_level_returns() -> None:
    selected = schedule_top_positions(_prediction_frame(), pool_name="all_open_known", max_positions=2)
    summary = summarize_portfolio(selected, pool_name="all_open_known", max_positions=2)

    assert summary["trade_count"] == 3
    assert summary["period_count"] == 2
    assert summary["mean_period_net_ret_pct"] == 1.25
