from __future__ import annotations

import pandas as pd

from quantlab.research.minute_ma_strategies import (
    attach_prior_daily_liquidity,
    build_liquidity_matched_control,
    build_liquidity_matched_controls_many,
    build_strategy_signals,
    build_strategy_signals_many,
    build_strategy_signals_vectorized,
    strategy_catalog,
)


def _state_rows() -> pd.DataFrame:
    rows = []
    closes = (10.2, 9.9, 10.1, 10.2, 10.3)
    lows = (10.1, 9.8, 10.0, 10.1, 10.2)
    highs = (10.3, 10.2, 10.2, 10.3, 10.4)
    for index, (close, low, high) in enumerate(zip(closes, lows, highs, strict=True)):
        rows.append(
            {
                "symbol": "A",
                "trade_date": "2022-01-03",
                "bar_time": f"09{31 + index:02d}00000",
                "sixty_minute_bucket": 1,
                "ma_period": 10,
                "hour_sequence": 0,
                "session_minute_ordinal": index,
                "adjusted_close": close,
                "adjusted_high": high,
                "adjusted_low": low,
                "amount": 100.0,
                "causal_intersection_adjusted": 10.0,
                "close_to_intersection_bps": (close / 10.0 - 1.0) * 10_000.0,
                "range_distance_to_intersection_bps": 0.0 if low <= 10.0 <= high else (low / 10.0 - 1.0) * 10_000.0,
                "touched_now": low <= 10.0 <= high,
                "close_below_intersection": close < 10.0,
                "previous_hour_close_adjusted": 10.0,
                "previous_hour_high_adjusted": 10.25,
                "prior_ma_slope_bps": 10.0,
                "ma_alignment_score": 1.0,
                "prior_true_touch_count_window": 0,
                "up_down_amount_ratio": 2.0,
                "bullish_ma_stack": True,
            }
        )
    return pd.DataFrame(rows)


def test_catalog_keeps_unavailable_layers_explicit() -> None:
    all_specs = strategy_catalog(include_unavailable=True)
    active_specs = strategy_catalog(include_unavailable=False)
    assert any(not spec.implemented for spec in all_specs)
    assert all(spec.implemented for spec in active_specs)


def test_random_control_is_stable_when_input_rows_are_reordered() -> None:
    states = _state_rows()
    first = build_strategy_signals(states, strategy_ids=["s0_random_matched"], random_seed=19)
    second = build_strategy_signals(
        states.sample(frac=1.0, random_state=3).reset_index(drop=True),
        strategy_ids=["s0_random_matched"],
        random_seed=19,
    )
    assert first["signal_time"].tolist() == second["signal_time"].tolist()


def test_reclaim_and_stable_rules_use_causal_confirmation_minute() -> None:
    states = _state_rows()
    reclaim = build_strategy_signals(states, strategy_ids=["s1_touch_reclaim"])
    stable = build_strategy_signals(states, strategy_ids=["s1_break_reclaim_stable3"])
    assert reclaim["signal_time"].tolist() == ["093300000"]
    assert stable["signal_time"].tolist() == ["093500000"]
    assert reclaim["causal_only"].all()
    assert stable["causal_only"].all()


def test_reclaim_can_confirm_an_intrabar_touch_that_closes_back_above() -> None:
    states = _state_rows().iloc[[0, 2, 3]].copy().reset_index(drop=True)
    states.loc[1, "touched_now"] = True
    states.loc[1, "close_below_intersection"] = False
    states.loc[1, "adjusted_low"] = 9.9
    result = build_strategy_signals(states, strategy_ids=["s1_touch_reclaim"])
    assert result["signal_time"].tolist() == ["093300000"]


def test_filtered_rule_can_use_a_later_reclaim_episode() -> None:
    states = _state_rows()
    # The first reclaim is deliberately rejected by the slope filter.  A
    # later break/reclaim is a separate episode and should still be tested.
    states.loc[2, "prior_ma_slope_bps"] = -5.0
    states.loc[3, "close_below_intersection"] = True
    states.loc[3, "touched_now"] = True
    states.loc[3, "adjusted_low"] = 9.8
    states.loc[4, "close_below_intersection"] = False
    states.loc[4, "touched_now"] = False
    states.loc[4, "prior_ma_slope_bps"] = 5.0
    result = build_strategy_signals(states, strategy_ids=["s2_reclaim_positive_slope"])
    assert result["signal_time"].tolist() == ["093500000"]


def test_posthoc_rule_is_marked_non_executable() -> None:
    spec = next(spec for spec in strategy_catalog() if spec.strategy_id == "s1_posthoc_catchup_diagnostic")
    assert spec.causal is False
    assert spec.executable is False
    assert spec.control is True


def test_batched_signal_builder_matches_individual_rules() -> None:
    states = _state_rows()
    ids = ["s0_random_matched", "s1_touch_reclaim", "s2_reclaim_positive_slope"]
    individual = pd.concat(
        [build_strategy_signals(states, strategy_ids=[strategy_id]) for strategy_id in ids],
        ignore_index=True,
    ).sort_values("signal_id", kind="stable").reset_index(drop=True)
    batched = build_strategy_signals_many(states, strategy_ids=ids).sort_values(
        "signal_id", kind="stable"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        individual,
        batched,
        check_dtype=False,
        check_like=True,
    )


def test_vectorized_signal_builder_matches_individual_rules() -> None:
    states = _state_rows()
    ids = ["s0_random_matched", "s1_touch_reclaim", "s1_near_reversal", "s2_reclaim_positive_slope"]
    individual = pd.concat(
        [build_strategy_signals(states, strategy_ids=[strategy_id]) for strategy_id in ids],
        ignore_index=True,
    ).sort_values("signal_id", kind="stable").reset_index(drop=True)
    vectorized = build_strategy_signals_vectorized(states, strategy_ids=ids).sort_values(
        "signal_id", kind="stable"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        individual,
        vectorized,
        check_dtype=False,
        check_like=True,
    )


def test_vectorized_filters_do_not_mutate_shared_rule_masks() -> None:
    states = _state_rows().assign(vwap_supportive=True)
    ids = [
        "s1_touch_reclaim",
        "s2_reclaim_positive_slope",
        "s4_reclaim_vwap_support",
    ]
    individual = pd.concat(
        [build_strategy_signals(states, strategy_ids=[strategy_id]) for strategy_id in ids],
        ignore_index=True,
    )
    vectorized = build_strategy_signals_vectorized(states, strategy_ids=ids)
    assert set(vectorized["strategy_id"]) == set(ids)
    assert len(vectorized) == len(individual)


def test_strong_control_uses_previous_high_not_previous_close() -> None:
    states = _state_rows()
    states.loc[:, "previous_hour_high_adjusted"] = 10.35
    result = build_strategy_signals(states, strategy_ids=["s0_strong_no_ma"])
    assert result.empty
    states.loc[states.index[-1], "adjusted_close"] = 10.4
    result = build_strategy_signals(states, strategy_ids=["s0_strong_no_ma"])
    assert result["signal_time"].tolist() == ["093500000"]


def _complete_daily_bars() -> pd.DataFrame:
    rows = []
    for symbol, amounts in (("A", [10.0, 20.0, 1000.0]), ("B", [30.0, 40.0, 2000.0])):
        for day_index, amount in enumerate(amounts):
            trade_date = f"2022-01-0{day_index + 1}"
            for ordinal in range(240):
                if ordinal < 120:
                    minute = ordinal
                    hour = 9 + (31 + minute) // 60
                    minute_value = (31 + minute) % 60
                else:
                    minute = ordinal - 120
                    hour = 13 + (1 + minute) // 60
                    minute_value = (1 + minute) % 60
                rows.append(
                    {
                        "symbol": symbol,
                        "trade_date": trade_date,
                        "bar_time": f"{hour:02d}{minute_value:02d}00000",
                        "amount": amount / 240.0,
                    }
                )
    return pd.DataFrame(rows)


def test_prior_liquidity_is_shifted_and_ignores_signal_day() -> None:
    states = _state_rows().assign(trade_date="2022-01-03")
    states = states.loc[states["bar_time"].eq("093100000")].copy()
    bars = _complete_daily_bars()
    enriched = attach_prior_daily_liquidity(states, bars, lookback_days=2)
    assert enriched["prior_20d_median_amount"].iloc[0] == 15.0


def test_liquidity_control_matches_same_minute_and_keeps_reference_id() -> None:
    base = _state_rows()
    other = base.copy()
    other["symbol"] = "B"
    other["prior_20d_median_amount"] = 110.0
    base["prior_20d_median_amount"] = 100.0
    states = pd.concat([base, other], ignore_index=True)
    reference = build_strategy_signals(states, strategy_ids=["s1_touch_reclaim"])
    reference = reference.loc[reference["symbol"].eq("A")].reset_index(drop=True)
    controls = build_liquidity_matched_control(states, reference)
    assert len(controls) == 1
    assert controls["symbol"].iloc[0] == "B"
    assert controls["reference_signal_id"].iloc[0] == reference["signal_id"].iloc[0]
    assert controls["reference_symbol"].iloc[0] == "A"
    assert controls["liquidity_match_ratio"].iloc[0] == 1.1


def test_many_liquidity_controls_clone_one_match_per_reference() -> None:
    base = _state_rows()
    other = base.copy()
    other["symbol"] = "B"
    base["prior_20d_median_amount"] = 100.0
    other["prior_20d_median_amount"] = 110.0
    states = pd.concat([base, other], ignore_index=True)
    first = build_strategy_signals(states, strategy_ids=["s1_touch_reclaim"])
    second = first.copy()
    second["strategy_id"] = "s2_reclaim_positive_slope"
    second["signal_id"] = second["signal_id"].str.replace("s1_touch_reclaim", "s2_reclaim_positive_slope", regex=False)
    references = pd.concat([first, second], ignore_index=True)
    controls = build_liquidity_matched_controls_many(states, references)
    assert len(controls) == 2
    assert set(controls["reference_signal_id"]) == set(references["signal_id"])
    assert controls["symbol"].eq("B").all()
