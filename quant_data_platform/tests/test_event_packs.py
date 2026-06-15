from __future__ import annotations

import pandas as pd

from quant_data_platform.event_packs.traditional_alpha import (
    add_derived_execution_labels,
    clean_qdp_feature_columns,
    is_execution_state_feature,
    parse_int_values,
    split_role_for_year,
)


def test_clean_qdp_feature_columns_drops_execution_state_only() -> None:
    columns = [
        "intraday_first_5m_ret",
        "holding_flag",
        "portfolio_cash_weight",
        "valuation_peTTM_cs_z",
        "recent_open_count_10d",
        "market_breadth_20",
    ]
    kept, dropped = clean_qdp_feature_columns(columns)

    assert kept == ["intraday_first_5m_ret", "valuation_peTTM_cs_z", "market_breadth_20"]
    assert dropped == ["holding_flag", "portfolio_cash_weight", "recent_open_count_10d"]
    assert is_execution_state_feature("last_action_is_open")
    assert not is_execution_state_feature("intraday_close_position")


def test_add_derived_execution_labels_uses_cost_and_limit_open_block() -> None:
    frame = pd.DataFrame(
        {
            "executable_entry": [True, False],
            "entry_open_near_limit": [False, True],
            "entry_one_word_limit": [False, False],
            "entry_day_close_ret_pct": [2.0, -1.0],
            "sell1_close_ret_pct": [3.0, -2.0],
            "sell1_max_high_pct": [5.0, 1.0],
            "sell1_min_low_pct": [-1.0, -6.0],
        }
    )

    out = add_derived_execution_labels(
        frame,
        sell_windows=(1,),
        fee_bps=30.0,
        slippage_bps=10.0,
        big_loss_threshold_pct=-5.0,
    )

    assert out["label_entry_tradeable_next_open"].tolist() == [True, False]
    assert out["label_entry_limit_up_buy_blocked"].tolist() == [False, True]
    assert out["label_ret_after_cost_d1_pct"].tolist() == [2.6, -2.4]
    assert out["label_big_loss_d1"].tolist() == [False, True]


def test_parse_int_values_and_split_roles() -> None:
    assert parse_int_values("1, 3,5") == (1, 3, 5)
    assert split_role_for_year(2023) == "train"
    assert split_role_for_year(2024) == "validation"
    assert split_role_for_year(2025) == "test"
