from __future__ import annotations

import json

import pandas as pd
import pytest

from traditional_quant_research.experiments import generalized_strong_event_pool_research as gen


def _event_input() -> pd.DataFrame:
    base = {
        "amount": 2.0e8,
        "signal_close_position": 0.80,
        "signal_amount_x20": 2.0,
        "signal_vrat5": 2.0,
        "limit_up_like": False,
        "close_cross_atr_upper": False,
        "high_cross_atr_upper": False,
        "new_high_20": False,
        "new_high_60": False,
        "open_below_kama_break_limitup": False,
        "ma_stack_bullish": False,
        "kama_slope_positive": False,
    }
    rows = [
        {**base, "code": "limit", "pctChg": 9.8, "limit_up_like": True, "amount": 5.0e7},
        {**base, "code": "near", "pctChg": 8.2},
        {**base, "code": "big", "pctChg": 6.0},
        {**base, "code": "atr", "pctChg": 3.5, "close_cross_atr_upper": True},
        {**base, "code": "high", "pctChg": 3.6, "new_high_60": True, "signal_amount_x20": 1.3},
        {**base, "code": "kama", "pctChg": 3.4, "open_below_kama_break_limitup": True, "signal_amount_x20": 1.3},
        {
            **base,
            "code": "trend",
            "pctChg": 4.2,
            "ma_stack_bullish": True,
            "kama_slope_positive": True,
        },
        {**base, "code": "illiquid", "pctChg": 6.5, "amount": 5.0e7},
    ]
    return pd.DataFrame(rows)


def test_assign_strong_event_columns_classifies_core_event_types() -> None:
    assigned = gen.assign_strong_event_columns(_event_input(), min_signal_amount=1.0e8)
    by_code = assigned.set_index("code")

    assert bool(by_code.loc["limit", "event_limit_up_core"]) is True
    assert by_code.loc["limit", "primary_event_type"] == "limit_up_core"
    assert bool(by_code.loc["near", "event_near_limit"]) is True
    assert by_code.loc["near", "primary_event_type"] == "near_limit"
    assert bool(by_code.loc["big", "event_big_up"]) is True
    assert bool(by_code.loc["atr", "event_volume_atr_breakout"]) is True
    assert bool(by_code.loc["high", "event_new_high_breakout"]) is True
    assert bool(by_code.loc["kama", "event_kama_breakout"]) is True
    assert bool(by_code.loc["trend", "event_trend_accel"]) is True
    assert bool(by_code.loc["illiquid", "strong_event_like"]) is False


def test_primary_event_type_priority_is_stable_for_overlapping_flags() -> None:
    frame = _event_input()
    frame.loc[frame["code"].eq("atr"), "new_high_60"] = True
    assigned = gen.assign_strong_event_columns(frame, min_signal_amount=1.0e8)
    row = assigned.loc[assigned["code"].eq("atr")].iloc[0]

    assert bool(row["event_volume_atr_breakout"]) is True
    assert bool(row["event_new_high_breakout"]) is True
    assert row["event_flag_count"] == 2
    assert row["primary_event_type"] == "volume_atr_breakout"


def test_build_pool_masks_splits_limit_and_executable_pools() -> None:
    assigned = gen.assign_strong_event_columns(_event_input(), min_signal_amount=1.0e8)
    assigned["entry_open_near_limit"] = [False, False, False, False, False, False, False, True]
    assigned["executable_entry"] = [True, True, True, True, True, True, False, True]
    masks = gen.build_pool_masks(assigned)

    assert int(masks["all_strong_events"].sum()) == len(assigned)
    assert int(masks["limit_up_core"].sum()) == 1
    assert int(masks["non_limit_strong"].sum()) == len(assigned) - 1
    assert int(masks["executable_only"].sum()) == 6
    assert int(masks["kama_breakout"].sum()) == 1


def test_summarize_event_pools_uses_target_and_execution_columns() -> None:
    assigned = gen.assign_strong_event_columns(_event_input(), min_signal_amount=1.0e8)
    assigned["target_net_ret_pct"] = [1.0, -2.0, 3.0, -6.0, 2.0, 4.0, -1.0, 9.0]
    assigned["entry_open_near_limit"] = False
    assigned["executable_entry"] = True

    summary = gen.summarize_event_pools(assigned)
    all_pool = summary.loc[summary["pool_name"].eq("all_strong_events")].iloc[0]
    kama_pool = summary.loc[summary["pool_name"].eq("kama_breakout")].iloc[0]

    assert all_pool["event_count"] == len(assigned)
    assert all_pool["big_loss_rate"] == pytest.approx(1.0 / len(assigned))
    assert kama_pool["event_count"] == 1
    assert kama_pool["mean_net_ret_pct"] == pytest.approx(4.0)


def test_enrich_predictions_with_event_columns_preserves_prediction_rows() -> None:
    assigned = gen.assign_strong_event_columns(_event_input(), min_signal_amount=1.0e8)
    assigned["date"] = pd.Timestamp("2024-01-02")
    predictions = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")],
            "code": ["near", "kama"],
            "ml_score": [2.0, 1.0],
        }
    )

    enriched = gen.enrich_predictions_with_event_columns(predictions, assigned)

    assert enriched["code"].tolist() == ["near", "kama"]
    assert enriched["primary_event_type"].tolist() == ["near_limit", "kama_breakout"]
    assert "event_strength_score" in enriched.columns


def test_cache_meta_matches_guards_parameter_changes(tmp_path) -> None:
    meta_path = tmp_path / "strong_events_2024.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "cache_version": gen.GENERALIZED_EVENT_CACHE_VERSION,
                "warmup_years": 1,
                "sell_windows": [1, 3],
                "min_signal_amount": 100000000.0,
            }
        ),
        encoding="utf-8",
    )

    assert gen.cache_meta_matches(meta_path, sell_windows=(1, 3), min_signal_amount=1.0e8, warmup_years=1)
    assert not gen.cache_meta_matches(meta_path, sell_windows=(1, 3, 5), min_signal_amount=1.0e8, warmup_years=1)
    assert not gen.cache_meta_matches(meta_path, sell_windows=(1, 3), min_signal_amount=2.0e8, warmup_years=1)
