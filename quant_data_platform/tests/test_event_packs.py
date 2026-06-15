from __future__ import annotations

import pandas as pd

from quant_data_platform.event_packs.traditional_alpha import (
    add_derived_execution_labels,
    apply_market_industry_context,
    build_market_industry_context,
    clean_qdp_feature_columns,
    event_candidate_codes,
    is_execution_state_feature,
    merge_quality_stats,
    new_quality_accumulator,
    parse_int_values,
    split_role_for_year,
    summarize_event_partition,
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


def test_quality_stats_are_incremental_without_retaining_frames() -> None:
    frame = pd.DataFrame(
        {
            "primary_event_type": ["limit_up_core", "big_up"],
            "split_role": ["train", "validation"],
            "qdp_feature_missing": [False, True],
            "qdp_a": [1.0, None],
            "qdp_b": [2.0, 3.0],
            "qdp_source_training_pack_manifest": ["m", "m"],
        }
    )

    stats = summarize_event_partition(frame)
    acc = new_quality_accumulator()
    merge_quality_stats(acc, stats)
    merge_quality_stats(acc, stats)

    assert stats["row_count"] == 2
    assert stats["qdp_feature_missing_rows"] == 1
    assert stats["qdp_feature_cells"] == 4
    assert stats["qdp_feature_nan_cells"] == 1
    assert acc["total_rows"] == 4
    assert acc["event_type_counts"]["limit_up_core"] == 2


def test_candidate_prefilter_keeps_event_eligible_codes_and_context() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02", "2023-12-29"]),
            "code": ["a", "b", "c", "b"],
            "industry": ["x", "x", "y", "x"],
            "high": [10.0, 10.0, 10.0, 9.0],
            "low": [9.0, 9.0, 9.0, 8.5],
            "close": [9.99, 9.8, 9.2, 8.8],
            "pctChg": [9.8, 4.0, 2.0, 5.0],
            "amount": [5.0e7, 2.0e8, 3.0e8, 2.0e8],
        }
    )

    assert event_candidate_codes(frame, year=2024, min_signal_amount=1.0e8) == {"a", "b"}

    context = build_market_industry_context(frame)
    events = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "code": ["b"],
            "industry": ["x"],
            "market_limitup_count": [-1],
            "industry_limitup_count": [-1],
        }
    )

    enriched = apply_market_industry_context(events, context)

    assert enriched.loc[0, "market_limitup_count"] == 1
    assert enriched.loc[0, "industry_limitup_count"] == 1
