from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy import seq100_strict_chan_parser as chan
from daily_research.path_policy import (
    seq100_strict_chan_structure_increment as study_module,
)


def test_frozen_sample_is_larger_and_excludes_prior_samples() -> None:
    study = study_module.load_study()
    spec, _ = study_module._sample_spec(study)
    assert sum(int(item["samples"]) for item in spec["selection"]["strata"]) == 700
    assert len(spec["selection"]["exclude_selected_cases"]) == 2
    assert spec["boundaries"]["return_test_performed"] is False


def test_pending_features_track_latest_candidate_state() -> None:
    def event(candidate: str, action: str, index: int) -> chan.SegmentStateEvent:
        return chan.SegmentStateEvent(
            id=f"{candidate}:{action}",
            candidate_id=candidate,
            action=action,
            direction=-1,
            segment_start_stroke_position=0,
            endpoint_stroke_position=3,
            middle_yin_stroke_id="stroke",
            break_case=2,
            event_index=index,
            event_time=f"2020-01-01 10:{index:02d}:00",
            confirmed_index=index,
            confirmed_time=f"2020-01-01 10:{index:02d}:00",
        )

    features = study_module._pending_features(
        [
            event("resolved", "opened", 1),
            event("resolved", "invalidated", 3),
            event("active", "opened", 4),
        ],
        confirmed_index=10,
    )
    assert features["active_pending_count_log"] == pytest.approx(np.log1p(1))
    assert features["active_pending_max_age_log"] == pytest.approx(np.log1p(6))
    assert features["recent_pending_invalidation_fraction"] == pytest.approx(1.0)


def test_complete_path_record_uses_next_open_and_exactly_40_sessions() -> None:
    study = study_module.load_study()
    dates = pd.bdate_range("2020-01-01", periods=42).strftime("%Y-%m-%d")
    close = np.linspace(10.0, 14.1, 42)
    daily = pd.DataFrame(
        {
            "trade_date": dates,
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "raw_open": close,
            "raw_close": close,
        }
    )
    case = {
        "case_id": "case",
        "symbol": "600000.SH",
        "stratum_id": "2020_2021_sh",
        "stratum_years": [2020, 2021],
        "focal_date": "2020-01-01",
    }
    record = study_module._path_record(
        case=case,
        episode_id=0,
        daily=daily,
        signal_index=0,
        maximum_sessions=40,
        costs=study["path"]["costs"],
    )
    assert record is not None
    entry = close[1]
    expected = close[1:41] / entry - 1.0
    assert record["entry_date"] == dates[1]
    assert record["exit_date"] == dates[40]
    assert record["mean_close_return_40"] == pytest.approx(expected.mean())
    assert record["terminal_close_return_40"] == pytest.approx(expected[-1])
    assert record["mfe_close_40"] == pytest.approx(expected.max())
    assert record["mae_close_40"] == pytest.approx(expected.min())
    assert record["terminal_net_return_base_40"] < expected[-1]


def test_match_controls_is_same_date_and_rejects_pattern_control() -> None:
    study = study_module.load_study()
    axes = list(study["matching"]["axes"])
    targets = list(study["evaluation"]["targets"])
    signal = {
        "signal_key": "signal",
        "case_id": "signal_case",
        "symbol": "600000.SH",
        "symbol_suffix": ".SH",
        "signal_date": "2020-01-02",
        "signal_year": 2020,
        "evaluation_split": "validation",
        "point_type": 1,
        "coordinate_available": True,
    }
    signal.update({axis: 0.50 for axis in axes})
    signal.update({target: 0.10 for target in targets})
    controls = []
    for key, symbol, value, has_point in (
        ("near_pattern", "600001.SH", 0.51, True),
        ("near_clean", "600002.SH", 0.52, False),
        ("far_clean", "600003.SH", 0.80, False),
    ):
        row = {
            "daily_key": key,
            "case_id": key,
            "symbol": symbol,
            "symbol_suffix": ".SH",
            "signal_date": "2020-01-02",
            "coordinate_available": True,
            "has_strict_buy_point": has_point,
        }
        row.update({axis: value for axis in axes})
        row.update({target: 0.0 for target in targets})
        controls.append(row)
    pairs = study_module._match_controls(
        pd.DataFrame([signal]), pd.DataFrame(controls), study
    )
    assert len(pairs) == 1
    assert pairs.iloc[0]["control_key"] == "near_clean"
    assert pairs.iloc[0][f"difference__{targets[0]}"] == pytest.approx(0.10)


def test_neighbor_predictions_use_only_prior_years() -> None:
    study = copy.deepcopy(study_module.load_study())
    study["evaluation"]["first_prediction_year"] = 2014
    study["evaluation"]["minimum_historical_signals_per_point_type"] = 2
    study["evaluation"]["neighbor_count"] = 2
    rows = []
    cases = (
        ("2012", 2012, 1.0, "2012-07-31"),
        ("2013", 2013, 2.0, "2013-07-31"),
        ("2013_unresolved", 2013, 50.0, "2014-12-31"),
        ("2014", 2014, 3.0, "2014-07-31"),
        ("2015", 2015, 100.0, "2015-07-31"),
    )
    for key, year, actual, exit_date in cases:
        row = {
            "signal_key": f"signal_{key}",
            "case_id": f"case_{key}",
            "symbol": f"600{len(rows):03d}.SH",
            "signal_date": f"{year}-06-01",
            "signal_year": year,
            "exit_date": exit_date,
            "evaluation_split": "development",
            "point_type": 1,
            "coordinate_available": True,
        }
        row.update({name: year / 10_000 for name in study["coordinates"]})
        row.update({name: year / 10_000 for name in study["structure_features"]})
        row.update({target: actual for target in study["evaluation"]["targets"]})
        rows.append(row)
    predictions = study_module._neighbor_predictions(pd.DataFrame(rows), study)
    selected = predictions[
        predictions["signal_year"].eq(2014)
        & predictions["target"].eq(study["evaluation"]["primary_target"])
    ]
    assert set(selected["representation"]) == {
        "coordinates",
        "structure",
        "combined",
    }
    assert selected["prediction"].tolist() == pytest.approx([1.5, 1.5, 1.5])
    assert selected["latest_historical_exit_date"].lt("2014-06-01").all()
