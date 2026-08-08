from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy import seq100_strict_chan_intraday as intraday
from daily_research.path_policy import seq100_strict_chan_parser as parser
from daily_research.path_policy import seq100_strict_chan_validate as validator


def _bars(length: int, seed: int = 20260808) -> pd.DataFrame:
    random = np.random.default_rng(seed)
    close = 20.0 * np.exp(np.cumsum(random.normal(0.0, 0.015, length)))
    spread = random.uniform(0.003, 0.02, length)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-02 09:35", periods=length, freq="5min"),
            "open": close,
            "high": close * np.exp(spread),
            "low": close * np.exp(-spread),
            "close": close,
            "volume": random.uniform(1.0e4, 1.0e6, length),
            "amount": random.uniform(1.0e6, 1.0e8, length),
        }
    )


def _stroke(
    position: int,
    direction: int,
    start_price: float,
    end_price: float,
) -> parser.Stroke:
    start_index = position * 3
    end_index = start_index + 2
    locked_index = end_index + 2
    start_time = pd.Timestamp("2020-01-02 09:35") + pd.Timedelta(
        minutes=5 * start_index
    )
    end_time = pd.Timestamp("2020-01-02 09:35") + pd.Timedelta(minutes=5 * end_index)
    locked_time = pd.Timestamp("2020-01-02 09:35") + pd.Timedelta(
        minutes=5 * locked_index
    )
    return parser.Stroke(
        id=f"stroke:{position}",
        direction=direction,
        start_fractal_id=f"fractal:{position}:start",
        end_fractal_id=f"fractal:{position}:end",
        start_normalized_position=position * 4,
        end_normalized_position=position * 4 + 3,
        start_index=start_index,
        end_index=end_index,
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat(),
        start_price=start_price,
        end_price=end_price,
        low=min(start_price, end_price),
        high=max(start_price, end_price),
        accepted_index=end_index + 1,
        accepted_time=(end_time + pd.Timedelta(minutes=5)).isoformat(),
        locked_index=locked_index,
        locked_time=locked_time.isoformat(),
        normalized_bar_span=4,
        raw_bar_span=3,
    )


def _segment(position: int, low: float, high: float, direction: int) -> parser.Segment:
    start_index = position * 10
    end_index = start_index + 9
    start_price = low if direction > 0 else high
    end_price = high if direction > 0 else low
    start_time = pd.Timestamp("2020-01-02") + pd.Timedelta(minutes=5 * start_index)
    end_time = pd.Timestamp("2020-01-02") + pd.Timedelta(minutes=5 * end_index)
    return parser.Segment(
        id=f"segment:{position}",
        direction=direction,
        start_stroke_position=position * 3,
        end_stroke_position=position * 3 + 2,
        stroke_ids=(f"s:{position}:0", f"s:{position}:1", f"s:{position}:2"),
        start_index=start_index,
        end_index=end_index,
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat(),
        start_price=start_price,
        end_price=end_price,
        low=low,
        high=high,
        event_index=end_index,
        event_time=end_time.isoformat(),
        confirmed_index=end_index + 3,
        confirmed_time=(end_time + pd.Timedelta(minutes=15)).isoformat(),
        break_case=1,
        feature_fractal_id=f"feature:{position}",
        raw_bar_span=10,
        price_change=float(np.log(end_price / start_price)),
        duration_normalized_price_change=abs(float(np.log(end_price / start_price)))
        / 10.0,
        macd_histogram_area=float(position + 1),
        amount_per_bar=1.0e7,
    )


def test_definition_contract_is_valid_and_keeps_profit_separate() -> None:
    spec = parser.load_definition_spec()
    assert len(spec["definitions"]) == 23
    assert all(item["implemented_in_parser_v1"] for item in spec["definitions"])
    assert spec["boundaries"]["replaces_weak_grammar"] is False
    assert spec["boundaries"]["profit_claim_allowed"] is False
    assert all(
        item["class"] == "empirical_hypothesis" and not item["parser_truth"]
        for item in spec["empirical_hypotheses"]
    )


def test_event_confirmation_never_precedes_event() -> None:
    result = parser.parse_strict_chan(_bars(1200))
    records = result.event_records()
    assert records
    assert all(
        int(record["confirmed_index"]) >= int(record["event_index"])
        for record in records
    )
    assert result.provisional_normalized_bar is not None
    assert all(
        current.confirmed_index >= previous.confirmed_index
        for previous, current in zip(result.segments, result.segments[1:], strict=False)
    )
    states: dict[str, list[parser.SegmentStateEvent]] = {}
    for event in result.segment_state_events:
        states.setdefault(event.candidate_id, []).append(event)
    for values in states.values():
        opened = next(item for item in values if item.action == "opened")
        assert all(item.confirmed_index >= opened.confirmed_index for item in values)


def test_parser_event_journal_is_prefix_invariant() -> None:
    result = validator.verify_prefix_invariance(
        _bars(700),
        cutoffs=[1, 4, 17, 80, 210, 430, 700],
    )
    assert result["status"] == "passed"
    assert result["cutoffs"] == 7
    assert result["future_confirmed_structure_written_back"] is False
    assert result["event_records_compared"] > 0


def test_initial_direction_lookahead_variant_handles_single_bar_prefix() -> None:
    result = validator.verify_prefix_invariance(
        _bars(300),
        cutoffs=[1, 2, 9, 80, 300],
        variant_overrides={
            "initial_inclusion_direction": "first_noninclusive_lookahead"
        },
    )
    assert result["status"] == "passed"


@pytest.mark.parametrize(
    "overrides",
    [
        {"inclusion_equality": "strict_containment"},
        {"fractal_equality": "outer_bar_tiebreak"},
        {"stroke_rule": "disjoint_fractals_only"},
        {"lowest_center_component": "stroke_diagnostic"},
        {"center_relation": "fluctuation_interval_overlap"},
    ],
)
def test_every_frozen_variant_is_prefix_invariant(
    overrides: dict[str, str],
) -> None:
    result = validator.verify_prefix_invariance(
        _bars(260),
        cutoffs=[1, 7, 40, 130, 260],
        variant_overrides=overrides,
    )
    assert result["status"] == "passed"


def test_segment_case_one_confirms_without_pending_state() -> None:
    strokes = [
        _stroke(0, 1, 1.0, 8.0),
        _stroke(1, -1, 8.0, 4.0),
        _stroke(2, 1, 4.0, 12.0),
        _stroke(3, -1, 12.0, 6.0),
        _stroke(4, 1, 6.0, 9.0),
        _stroke(5, -1, 9.0, 2.0),
        _stroke(6, 1, 2.0, 8.0),
        _stroke(7, -1, 8.0, 1.0),
    ]
    bars = _bars(40)
    segments, events = parser.build_segments(strokes, bars)
    assert segments
    assert segments[0].direction == 1
    assert segments[0].break_case == 1
    assert segments[0].stroke_ids == ("stroke:0", "stroke:1", "stroke:2")
    assert not events


def test_segment_case_two_stays_pending_then_confirms() -> None:
    opening = [
        _stroke(0, 1, 1.0, 8.0),
        _stroke(1, -1, 8.0, 4.0),
        _stroke(2, 1, 4.0, 12.0),
        _stroke(3, -1, 12.0, 10.0),
        _stroke(4, 1, 10.0, 11.0),
        _stroke(5, -1, 11.0, 2.0),
        _stroke(6, 1, 2.0, 7.0),
        _stroke(7, -1, 7.0, 1.0),
    ]
    pending_segments, pending_events = parser.build_segments(opening, _bars(80))
    assert not pending_segments
    assert [item.action for item in pending_events] == ["opened"]
    completed = opening + [
        _stroke(8, 1, 3.0, 9.0),
        _stroke(9, -1, 9.0, 4.0),
        _stroke(10, 1, 4.0, 10.0),
        _stroke(11, -1, 10.0, 3.0),
    ]
    segments, events = parser.build_segments(completed, _bars(100))
    assert segments
    assert segments[0].break_case == 2
    assert [item.action for item in events[:2]] == ["opened", "confirmed"]
    assert events[0].middle_yin_stroke_id == "stroke:3"
    assert events[1].confirmed_index > events[0].confirmed_index


def test_case_two_invalidation_cannot_precede_candidate_detection() -> None:
    candidate = parser._FeatureFractalCandidate(
        id="feature:late",
        direction=1,
        endpoint_stroke_position=0,
        endpoint_price=5.0,
        event_index=4,
        event_time="2020-01-02T09:55:00",
        detected_index=50,
        detected_time="2020-01-02T13:45:00",
        break_case=2,
    )
    invalidation = parser._candidate_invalidation(
        [_stroke(0, -1, 8.0, 4.0), _stroke(1, 1, 4.0, 9.0)],
        candidate,
    )
    assert invalidation == (50, "2020-01-02T13:45:00")


def test_recursive_centers_are_built_from_completed_movements() -> None:
    intervals = [
        (1.0, 20.0),
        (4.0, 6.0),
        (3.0, 7.0),
        (7.0, 12.0),
        (1.0, 22.0),
        (8.0, 10.0),
        (11.0, 16.0),
        (1.0, 24.0),
        (12.0, 14.0),
        (15.0, 20.0),
        (1.0, 26.0),
        (16.0, 18.0),
    ]
    segments = [
        _segment(position, low, high, 1 if position % 2 == 0 else -1)
        for position, (low, high) in enumerate(intervals)
    ]
    centers, _, _, movements = parser.build_recursive_centers(
        segments,
        [],
        base_component="segment",
        relation_mode="core_interval_overlap",
    )
    assert sum(item.level == 1 for item in centers) == 4
    assert sum(item.level == 1 for item in movements) == 3
    assert any(
        item.level == 2 and item.component_kind == "movement" for item in centers
    )


def test_intraday_completeness_and_missing_days_split_episodes() -> None:
    complete = pd.DataFrame(
        {
            "symbol": ["600000.SH"] * 48,
            "trade_date": ["2025-01-02"] * 48,
            "bar_time": list(intraday.EXPECTED_5M_TIMES),
            "open": np.full(48, 10.0),
            "high": np.full(48, 10.1),
            "low": np.full(48, 9.9),
            "close": np.full(48, 10.0),
        }
    )
    incomplete = complete.iloc[:-1].copy()
    incomplete["trade_date"] = "2025-01-03"
    dates, audit = intraday._complete_intraday_dates(
        pd.concat([complete, incomplete], ignore_index=True)
    )
    assert dates == {"2025-01-02"}
    assert audit["invalid_day_examples"][0]["trade_date"] == "2025-01-03"
    episodes = intraday._assign_episode_ids(
        ["2025-01-02", "2025-01-06", "2025-01-07"],
        {"2025-01-03"},
    )
    assert episodes == {
        "2025-01-02": 0,
        "2025-01-06": 1,
        "2025-01-07": 1,
    }
